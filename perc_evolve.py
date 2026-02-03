import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from balrog.agents import AgentFactory
from balrog.evaluator import EvaluatorManager
from balrog.utils import collect_and_summarize_results, print_summary_table, setup_environment
from improve import improve_both, improve_perception_only

# Top-level logger for high-level evolution logs (eval.log)
evolve_logger = logging.getLogger("evolve")


@contextmanager
def step_logging(step_output_dir: Path):
    """Context manager to redirect all logging to a step-specific log file.

    During the context, all log messages go to step_output_dir/step.log.
    After exiting, logging returns to normal (eval.log).
    """
    step_log_file = step_output_dir / "step.log"

    # Create a handler for the step log
    step_handler = logging.FileHandler(step_log_file)
    step_handler.setLevel(logging.INFO)
    step_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Get the root logger and save its current handlers
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()

    # Replace handlers with step handler
    root_logger.handlers = [step_handler]

    try:
        yield step_log_file
    finally:
        # Restore original handlers
        step_handler.close()
        root_logger.handlers = original_handlers


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
    config.eval.perception_prompt = perception
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
    instructions from config.eval.instruction_path.
    """
    # Load perception module
    if (perception_path := config.eval.get("perception_path", None)) is None:
        logging.info("Path to perception not specified as eval.perception_path arg, using empty perception")
        perception = ""
    else:
        perception = Path(perception_path).read_text()
        logging.info(f"Loaded perception module from: {perception_path}")

    # Load instructions/beliefs
    if (instruction_path := config.eval.get("instruction_path", None)) is None:
        logging.info("Path to instructions not specified as eval.instruction_path arg, using empty instruction")
        instruction = ""
    else:
        instruction = Path(instruction_path).read_text()
        logging.info(f"Loaded instructions from: {instruction_path}")

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


@dataclass
class EvolveConfig:
    num_steps: int
    rollouts_per_step: int


def find_last_completed_step(output_dir: str) -> tuple[int, str, str]:
    """Find the last completed step with both beliefs.txt and perception.py files.

    Args:
        output_dir: Output directory containing step folders

    Returns:
        Tuple of (last_step_number, beliefs_content, perception_content).
        Returns (0, "", "") if no steps found.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0, "", ""

    # Find all step directories
    step_dirs = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            try:
                step_num = int(item.name.split("_")[1])
                beliefs_file = item / "beliefs.txt"
                perception_file = item / "perception.py"
                if beliefs_file.exists() and perception_file.exists():
                    step_dirs.append((step_num, beliefs_file, perception_file))
            except (ValueError, IndexError):
                continue

    if not step_dirs:
        return 0, "", ""

    # Sort by step number and get the last one
    step_dirs.sort(key=lambda x: x[0])
    last_step_num, last_beliefs_file, last_perception_file = step_dirs[-1]
    beliefs_content = last_beliefs_file.read_text()
    perception_content = last_perception_file.read_text()

    evolve_logger.info(f"Found last completed step: {last_step_num}")
    evolve_logger.info(f"Resuming with beliefs from: {last_beliefs_file}")
    evolve_logger.info(f"Resuming with perception from: {last_perception_file}")

    return last_step_num, beliefs_content, perception_content


def online_evolve(
    evolve_config: EvolveConfig,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    """Run online evolution loop for beliefs and/or perception.

    Args:
        evolve_config: Evolution configuration
        config: BALROG configuration
        original_cwd: Original working directory
        output_dir: Output directory for results

    Note:
        The improve mode is controlled by config.eval.evolve.improve_mode:
        - "perception_only": Only evolve perception, beliefs stay fixed
        - "both": Evolve both beliefs and perception (default)

        If config.eval.instruction_path is set, initial beliefs are loaded from that file.

    Logging:
        - High-level logs (performance, improved H/P) go to eval.log
        - Detailed step logs (agent calls, trajectories) go to step_X/step.log
    """
    improve_mode = config.eval.evolve.get("improve_mode", "both")
    evolve_logger.info(f"Running evolution with improve_mode: {improve_mode}")

    # Check for existing progress and resume if available
    last_step, h, p = find_last_completed_step(output_dir)
    start_step = last_step + 1

    if last_step > 0:
        evolve_logger.info(f"Resuming from step {start_step} (found {last_step} completed steps)")
    else:
        evolve_logger.info("Starting fresh evolution (no existing steps found)")
        # Load initial beliefs from file if specified
        if (instruction_path := config.eval.get("instruction_path", None)) is not None:
            h = Path(instruction_path).read_text()
            evolve_logger.info(f"Loaded initial beliefs from: {instruction_path}")
        else:
            h = ""
        
        if (perception_path := config.eval.get("perception_path", None)) is not None:
            p = Path(perception_path).read_text()
            evolve_logger.info(f"Loaded initial perception from: {instruction_path}")
        else:
            p = ""

    config.eval.num_episodes = evolve_config.rollouts_per_step

    evolve_logger.info(f"Starting online evolution with {evolve_config.num_steps} steps")
    evolve_logger.info(f"Rollouts per step: {evolve_config.rollouts_per_step}")

    for step in range(start_step, evolve_config.num_steps + 1):
        evolve_logger.info(f"\n{'='*80}")
        evolve_logger.info(f"STEP {step}/{evolve_config.num_steps}")
        evolve_logger.info(f"{'='*80}")

        step_output_dir = Path(output_dir) / f"step_{step}"
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Run rollouts and improvement with step-specific logging
        with step_logging(step_output_dir) as step_log_file:
            logging.info(f"Step {step} detailed logs")
            logging.info(f"Current beliefs:\n{h if h else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")

            # Run rollouts with current beliefs and perception
            step_summary = one_step(
                instruction=h,
                perception=p,
                config=config,
                original_cwd=original_cwd,
                output_dir=str(step_output_dir),
            )

            logging.info(f"Step {step} summary: {step_summary}")

            # Save step summary to JSON
            summary_file = step_output_dir / "summary.json"
            with open(summary_file, "w") as f:
                json.dump(step_summary, f, indent=4)
            logging.info(f"Saved step summary to {summary_file}")

            # Select improve function based on config
            improve_mode = config.eval.evolve.get("improve_mode", "both")
            synthesis_prompt = None
            match improve_mode:
                case "perception_only":
                    p, synthesis_prompt = improve_perception_only(config, h, p, str(step_output_dir))
                case "both":
                    h, p, synthesis_prompt = improve_both(config, h, p, str(step_output_dir))
                case _:
                    logging.error(f"Unknown improve_mode: {improve_mode}. Using 'both' as default.")
                    h, p, synthesis_prompt = improve_both(config, h, p, str(step_output_dir))

            # Save beliefs
            beliefs_file = step_output_dir / "beliefs.txt"
            beliefs_file.write_text(h)
            logging.info(f"Saved beliefs to {beliefs_file}")

            # Save updated perception module
            perception_file = step_output_dir / "perception.py"
            perception_file.write_text(p)
            logging.info(f"Saved updated perception to {perception_file}")

        # Log high-level summary to eval.log
        evolve_logger.info(f"Step {step} performance: {step_summary}")
        if synthesis_prompt:
            evolve_logger.info(f"Step {step} synthesis prompt:\n{synthesis_prompt}")
        evolve_logger.info(f"Step {step} updated beliefs:\n{h}")
        evolve_logger.info(f"Step {step} updated perception:\n{p}")


@contextmanager
def redirect_to_file(filepath):
    original = sys.stdout
    with open(filepath, "w") as file:
        sys.stdout = file
        try:
            yield
        finally:
            sys.stdout = original


@hydra.main(config_path="balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    original_cwd = get_original_cwd()
    setup_environment(original_cwd=original_cwd)

    # Determine output directory
    if config.eval.resume_from is not None:
        output_dir: str = config.eval.resume_from
    else:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{config.agent.type}_{config.client.model_id.replace('/', '_')}_perc"
        output_dir = os.path.join(config.eval.output_dir, run_name)

        # Create the directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup loggers
    log_filename = os.path.join(output_dir, "eval.log")
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Configure root logger (used for detailed step logs)
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.FileHandler(log_filename)],
        force=True,
    )

    # Configure evolve_logger for high-level logs (always writes to eval.log)
    evolve_handler = logging.FileHandler(log_filename)
    evolve_handler.setLevel(logging.INFO)
    evolve_handler.setFormatter(logging.Formatter(log_format))
    evolve_logger.addHandler(evolve_handler)
    evolve_logger.setLevel(logging.INFO)
    evolve_logger.propagate = False  # Don't propagate to root logger

    # Print output location to terminal
    print(f"Output directory: {output_dir}")
    print(f"Log file: {log_filename}")

    # Save config to output directory
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config=config, f=f)
    evolve_logger.info(f"Saved config to {config_path}")


    match config.eval.mode:
        case "eval":
            one_step_wrap(config=config, original_cwd=original_cwd, output_dir=output_dir)

        case "evolve":
            ec = EvolveConfig(
                num_steps=config.eval.evolve.num_steps,
                rollouts_per_step=config.eval.evolve.rollouts_per_step,
            )
            online_evolve(
                evolve_config=ec,
                config=config,
                original_cwd=original_cwd,
                output_dir=output_dir,
            )
        case _:
            evolve_logger.error(f"Unsupported mode: {config.eval.mode}. perc_evolve.py supports 'eval' and 'evolve' modes.")
            raise ValueError(f"Unsupported mode: {config.eval.mode}")


if __name__ == "__main__":
    main()
