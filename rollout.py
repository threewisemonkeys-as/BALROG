import concurrent.futures
import json
import logging
import os
from pathlib import Path

from omegaconf import DictConfig

from balrog.agents import AgentFactory
from balrog.evaluator import EvaluatorManager


def summarise_results(results: dict):
    summary = {}
    for env_name, env_results in results.items():
        num_attempts = len(env_results)
        summary[env_name] = {
            "num_perfect": sum([r['progression'] == 1.0 for r in env_results]),
            "num_solved": sum([r['progression'] > 0.8 for r in env_results]),
            "avg_prog": sum([r['progression'] for r in env_results]) / num_attempts,
            "avg_steps": sum([r['num_steps'] for r in env_results]) / num_attempts,
            "total_cost": sum([r['total_cost'] for r in env_results]),
            "avg_cost": sum([r['total_cost'] for r in env_results]) / num_attempts,
        }

    return summary


def one_step(
    instruction: str,
    perception: str,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    """Run one step with both instruction and perception module."""
    config.eval.instruction_prompt = instruction

    # Write perception to file to avoid OmegaConf parsing issues with special characters
    # (OmegaConf interprets '${' as interpolation syntax, which can appear in perception code)
    if perception:
        perception_file = Path(output_dir) / "_perception_module.py"
        perception_file.write_text(perception)
        config.eval.perception_path = str(perception_file)
    else:
        config.eval.perception_path = None

    evaluator_manager = EvaluatorManager(config, original_cwd=original_cwd, output_dir=output_dir)
    agent_factory = AgentFactory(config)
    results = evaluator_manager.run(agent_factory)
    summary = summarise_results(results)
    return summary


def one_step_wrap(
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    """Run evaluation with perception and instruction loaded from files.

    Loads perception from config.eval.perception_path and optionally
    instructions from config.eval.beliefs_path.
    """
    # Load perception module
    if (perception_path := config.eval.get("perception_path", None)) is None:
        logging.info("Path to perception not specified as eval.perception_path arg, using empty perception")
        perception = ""
    else:
        perception = Path(perception_path).read_text()
        logging.info(f"Loaded perception module from: {perception_path}")

    # Load instructions/beliefs
    if (beliefs_path := config.eval.get("beliefs_path", None)) is None:
        logging.info("Path to instructions not specified as eval.beliefs_path arg, using empty instruction")
        instruction = ""
    else:
        instruction = Path(beliefs_path).read_text()
        logging.info(f"Loaded instructions from: {beliefs_path}")

    logging.info(f"Using following as instruction -\n{instruction}")
    logging.info(f"Using following as perception -\n{perception}")

    summary = one_step(
        instruction=instruction,
        perception=perception,
        config=config,
        original_cwd=original_cwd,
        output_dir=output_dir,
    )

    print(summary)
    json.dump(summary, open(Path(output_dir) / "summary.json", "w"), indent=4)


def run_single_rollout_task(
    run_name: str,
    base_beliefs: str,
    perception: str,
    experiment: str | None,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
) -> tuple[str, dict]:
    """Helper function to run a single rollout task in a separate process.

    Args:
        run_name: Name of the run (e.g., baseline_0, experiment_1)
        base_beliefs: Base beliefs
        perception: Current perception module
        experiment: Specific experiment to test (or None for baseline)
        config: Configuration object
        original_cwd: Original working directory
        output_dir: Root output directory

    Returns:
        Tuple of (run_name, result_dict)
    """
    # Force single worker per process to avoid nested pools
    config.eval.num_workers = 1

    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    instruction_to_use = base_beliefs
    result_type = "baseline"

    if experiment:
        result_type = "experiment"
        # Save the experiment being tested
        (run_dir / "experiment.txt").write_text(experiment)

        # Combine base instruction with experiment
        if base_beliefs:
            instruction_to_use = f"""{base_beliefs}

=== EXPERIMENT TO TEST ===
The following experiment might help achieve the goal. Try to test it during gameplay:

{experiment}
=== END EXPERIMENT ===
"""
        else:
            instruction_to_use = f"""=== EXPERIMENT TO TEST ===
The following experiment might help achieve the goal. Try to test it during gameplay:

{experiment}
=== END EXPERIMENT ===
"""

    logging.info(f"Starting rollout for {run_name} in process {os.getpid()}")

    try:
        summary = one_step(
            instruction=instruction_to_use,
            perception=perception,
            config=config,
            original_cwd=original_cwd,
            output_dir=str(run_dir),
        )

        result_data = {
            "type": result_type,
            "summary": summary,
        }
        if experiment:
            result_data["experiment"] = experiment

        return run_name, result_data

    except Exception as e:
        logging.error(f"Rollout {run_name} failed: {e}")
        return run_name, {"type": result_type, "error": str(e)}


def run_explore_rollouts(
    base_beliefs: str,
    perception: str,
    experiments: list[str],
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
    num_baseline_rollouts: int = 3,
) -> dict[str, dict]:
    """Run rollouts for exploration in parallel using ProcessPoolExecutor.

    If experiments are provided, runs one rollout per experiment.
    If no experiments are provided (empty list), runs `num_baseline_rollouts` baseline rollouts.

    Args:
        base_beliefs: Base beliefs/instructions
        perception: Current perception module
        experiments: List of experiment strings to test (can be empty)
        config: BALROG configuration
        original_cwd: Original working directory
        output_dir: Output directory for results
        num_baseline_rollouts: Number of rollouts to run if no experiments are provided

    Returns:
        Dictionary mapping run index/name to their results summary
    """
    all_results = {}

    # Determine tasks to run
    tasks = []
    if not experiments:
        logging.info(f"No experiments provided. Preparing {num_baseline_rollouts} baseline rollouts.")
        for i in range(num_baseline_rollouts):
            tasks.append({
                "run_name": f"baseline_{i}",
                "experiment": None
            })
    else:
        logging.info(f"Preparing {len(experiments)} experiment rollouts.")
        for i, experiment in enumerate(experiments):
            tasks.append({
                "run_name": f"experiment_{i}",
                "experiment": experiment
            })

    # Use configured number of workers for parallelism
    max_workers = config.eval.num_workers
    logging.info(f"Running rollouts in parallel with max_workers={max_workers}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task in tasks:
            future = executor.submit(
                run_single_rollout_task,
                run_name=task["run_name"],
                base_beliefs=base_beliefs,
                perception=perception,
                experiment=task["experiment"],
                config=config,
                original_cwd=original_cwd,
                output_dir=output_dir
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                run_name, result = future.result()
                all_results[run_name] = result
                logging.info(f"Completed rollout for {run_name}")
            except Exception as e:
                logging.error(f"Worker execution failed: {e}")

    return all_results
