import asyncio
import concurrent.futures
import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hydra
import litellm
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from balrog.agents import AgentFactory
from balrog.evaluator import EvaluatorManager
from balrog.utils import setup_environment
from balrog.environments import make_env
from balrog.environments.minihack import get_loaded_instruction_prompt
from improve import get_episode_summary_async, trim_to_model_context_lim, _get_response_cost
from llm_utils import build_llm_input, extract_llm_response_text, extract_xml_kv, validate_response_fields
from explore import (
    step_logging,
    improve_logging,
    get_default_knowledge,
    ExploreConfig,
)
from improve import improve_step, validate_perception_code
from rollout import summarise_results, one_step, run_single_rollout_task, run_explore_rollouts

# Top-level logger for high-level evolution logs (eval.log)
evolve_logger = logging.getLogger("evolve")


def improve_step_no_experiments(
    config: DictConfig,
    base_beliefs: str,
    perception: str,
    output_dir: str,
    previous_experiments: list[str],
    default_knowledge: str,
    rollout_results: dict[str, dict] | None = None,
) -> tuple[str, str, float]:
    """Improve step that evaluates experiments but does NOT generate new ones.

    Analyses rollout results (including experiment evaluation if experiments were
    tested), updates beliefs and perception. No new experiments are generated.

    Args:
        config: Configuration containing model information
        base_beliefs: Current beliefs/instructions
        perception: Current perception module
        output_dir: Directory containing rollout results
        previous_experiments: List of experiments tested in the rollouts
        default_knowledge: Default knowledge string to include in prompt
        rollout_results: Results from run_explore_rollouts, including any errors

    Returns:
        Tuple of (updated_beliefs, updated_perception, total_improve_cost)
    """

    async def get_all_summaries():
        """Get all episode summaries in parallel."""
        episode_tasks = []

        for episode_path in Path(output_dir).rglob("*.csv"):
            try:
                # Get path relative to output_dir to determine run name
                rel_path = episode_path.relative_to(output_dir)
                run_name = rel_path.parts[0]  # e.g., baseline_0, hypothesis_1

                experiment_text = None
                if run_name.startswith("experiment_"):
                    try:
                        idx = int(run_name.split("_")[1])
                        if 0 <= idx < len(previous_experiments):
                            experiment_text = previous_experiments[idx]
                    except (ValueError, IndexError):
                        logging.warning(f"Could not map run {run_name} to an experiment")

                episode_tasks.append(
                    get_episode_summary_async(
                        config,
                        base_beliefs,
                        perception,
                        str(episode_path),
                        experiment=experiment_text,
                    )
                )
            except ValueError:
                logging.warning(f"Could not determine run name for {episode_path}")
                continue

        logging.info(f"Generating summaries for {len(episode_tasks)} episodes in parallel (no experiment generation)")
        return await asyncio.gather(*episode_tasks)

    # Get summaries for all episodes in parallel
    ep_results = asyncio.run(get_all_summaries())

    # Combine all summaries and track costs
    evidence_section = ""
    summary_cost = 0.0
    for i, (summary, cost) in enumerate(ep_results):
        evidence_section += f"Episode {i+1} Summary:\n{summary.strip()}\n\n"
        summary_cost += cost

    # Include any rollout errors in evidence section
    if rollout_results:
        for run_name, result in rollout_results.items():
            if "error" in result:
                evidence_section += f"Rollout {run_name} ERROR: {result['error']}\n\n"

    base_prompt = f"""We are playing a game and trying to figure out how it works.
Current beliefs about the game:
{base_beliefs if base_beliefs else "(empty - no beliefs yet)"}

The agent also receives the following default instructions/knowledge by default:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

Current perception module:
{perception if perception else "(empty - no perception module yet)"}

We have collected new experience by attempting to play the game with certain experiments in mind.
{evidence_section}

Your task is to:
1. Analyze the results. 
2. Update our beliefs about the game based on confirmed knowledge.
3. Update the perception module to make sure it is correct and that it extracts better features from the direct game observation.

For beliefs:
- They should describe essential information about how the game works.
- They should be very brief with a limit of 10 points, each of which should be only a few sentences. It is important to keep the beliefs simple.
- Correct any wrong or misleading beliefs
- From evidence present from the trajectories, infer beliefs that might lead to a positive outcome.

For the perception module:
- It should be a Python function `perceive(observation_text: str) -> str`.
- Input `observation_text` contains everything from the direct game observation as a string.
- Ensure that the perception module is working correctly in that it is correctly extracting the intended information from the direct game state and presenting it in the features from perception module section.
- Output should be a textual description of the game state that is useful for progressing in the game.
- The output should be brief and to the point.
- The code must be valid Python.

It is important that you keep both the beliefs and the output of the perception module as simple as possible.
Since we are only evaluating experiment, only update the beliefs or the perception if we have learned anything new from the collected experience. Do not update them if we have not learned anything new.

Format your response in XML style as:
<think>
Analyze results, evaluate experiments, determine belief updates, and design perception improvements.
</think>
<updated_beliefs>
- [belief 1]
- [belief 2]
...
</updated_beliefs>
<perception>
```python
def perceive(observation_text: str) -> str:
    # Your implementation here
    pass
```
</perception>
"""

    # Setup model name
    if config.client.client_name == "vllm":
        model_name = f"hosted_vllm/{config.client.model_id}"
    else:
        model_name = f"{config.client.client_name}/{config.client.model_id}"

    # Retry loop for perception validation
    max_retries = 3
    perception_error = None
    updated_beliefs = base_beliefs
    updated_perception = perception
    improve_call_cost = 0.0

    for attempt in range(max_retries):
        # Build prompt with error feedback if this is a retry
        if perception_error:
            prompt = f"""{base_prompt}

=== PERCEPTION CODE ERROR (RETRY {attempt}/{max_retries}) ===
Your previous perception code had an error and failed to execute:
{perception_error}

Please fix the error in your perception code.
=== END ERROR ===
"""
        else:
            prompt = base_prompt

        logging.info(f"Baseline improve step prompt (attempt {attempt + 1}/{max_retries}):\n{prompt}")

        # Build input for LLM
        input_data = build_llm_input(prompt)

        # Call LLM
        logging.info(f"Calling LLM for baseline improve step (attempt {attempt + 1}/{max_retries})")
        response = litellm.responses(
            model=model_name,
            input=input_data,
            num_retries=5,
        )
        improve_call_cost += _get_response_cost(response, config.client.model_id)

        # Extract response text
        response_text = extract_llm_response_text(response)
        logging.info(f"Baseline improve step LLM response (attempt {attempt + 1}/{max_retries}):\n{response_text}")

        # Extract fields
        response_dict = extract_xml_kv(response_text, ["updated_beliefs", "perception"])
        validate_response_fields(response_dict, response_text, ["updated_beliefs", "perception"])

        # Process beliefs
        if "updated_beliefs" in response_dict:
            updated_beliefs = response_dict["updated_beliefs"].strip()

        # Process perception
        candidate_perception = perception
        if "perception" in response_dict:
            candidate_perception = response_dict["perception"].strip()
            # Strip markdown markers
            if candidate_perception.startswith("```python"):
                candidate_perception = candidate_perception[len("```python"):].strip()
            elif candidate_perception.startswith("```"):
                candidate_perception = candidate_perception[len("```"):].strip()
            if candidate_perception.endswith("```"):
                candidate_perception = candidate_perception[:-len("```")].strip()

        # Validate perception code
        is_valid, error_msg = validate_perception_code(candidate_perception)
        if is_valid:
            updated_perception = candidate_perception
            perception_error = None
            logging.info(f"Perception code validated successfully on attempt {attempt + 1}")
            break
        else:
            perception_error = error_msg
            logging.warning(f"Perception code validation failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
            if attempt == max_retries - 1:
                logging.error(f"All {max_retries} attempts to generate valid perception code failed. Keeping previous perception.")
                updated_perception = perception

    logging.info(f"Updated beliefs (baseline):\n{updated_beliefs}")
    logging.info(f"Updated perception (baseline):\n{updated_perception}")

    total_improve_cost = summary_cost + improve_call_cost
    return updated_beliefs, updated_perception, total_improve_cost


def find_last_completed_step(output_dir: str) -> tuple[int, str, str]:
    """Find the last completed step with beliefs.txt and perception.py files.

    Looks for the explore phase output (which is the final phase of each step).
    Experiments are not carried across steps since Phase A generates them fresh.

    Args:
        output_dir: Output directory containing step folders

    Returns:
        Tuple of (last_step_number, beliefs_content, perception_content).
        Returns (0, "", "") if no steps found.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0, "", ""

    # Find all step directories that have a completed explore phase
    step_dirs = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            try:
                step_num = int(item.name.split("_")[1])
                # Check for explore phase completion (final output of each step)
                explore_dir = item / "explore"
                beliefs_file = explore_dir / "beliefs.txt"
                perception_file = explore_dir / "perception.py"
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

    # Hypotheses are not carried across steps (Phase A generates them fresh each step)

    evolve_logger.info(f"Found last completed step: {last_step_num}")
    evolve_logger.info(f"Resuming with beliefs from: {last_beliefs_file}")
    evolve_logger.info(f"Resuming with perception from: {last_perception_file}")

    return last_step_num, beliefs_content, perception_content


def _log_rollout_stats(rollout_results: dict[str, dict], phase_name: str):
    """Calculate and log rollout success statistics."""
    total_rollouts = len(rollout_results)
    successful_rollouts = 0
    partial_rollouts = 0
    failed_rollouts = 0
    error_rollouts = 0

    for run_name, result in rollout_results.items():
        if "error" in result:
            error_rollouts += 1
            continue
        summary = result.get("summary", {})
        for env_name, env_stats in summary.items():
            if env_stats.get("num_perfect", 0) > 0:
                successful_rollouts += 1
            elif env_stats.get("avg_prog", 0) > 0:
                partial_rollouts += 1
            else:
                failed_rollouts += 1

    evolve_logger.info(
        f"[{phase_name}] Rollout results: {total_rollouts} total, {successful_rollouts} successful (100%), "
        f"{partial_rollouts} partial progress, {failed_rollouts} failed (0%), {error_rollouts} errors"
    )


def _compute_rollout_cost(rollout_results: dict[str, dict]) -> float:
    """Sum up rollout costs from results."""
    cost = 0.0
    for _, result in rollout_results.items():
        if "summary" in result:
            for env_stats in result["summary"].values():
                cost += env_stats.get("total_cost", 0)
    return cost


def online_explore_2step(
    explore_config: ExploreConfig,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    """Run 2-step online exploration loop.

    Each iteration consists of:
      Phase A (Baseline): Rollouts without hypotheses -> Improve (updates beliefs/perception, generates hypotheses)
      Phase B (Explore):  Rollouts with hypotheses   -> Improve (evaluates hypotheses, updates beliefs/perception, no new hypotheses)

    The baseline phase focuses on pure gameplay and generates hypotheses to test.
    The explore phase tests those hypotheses and incorporates findings.

    Args:
        explore_config: Exploration configuration
        config: BALROG configuration
        original_cwd: Original working directory
        output_dir: Output directory for results
    """
    evolve_logger.info("Running 2-step exploration (baseline + explore per iteration)")

    # Check for existing progress and resume if available
    last_step, h, p = find_last_completed_step(output_dir)
    start_step = last_step + 1

    if last_step > 0:
        evolve_logger.info(f"Resuming from step {start_step} (found {last_step} completed steps)")
    else:
        evolve_logger.info("Starting fresh 2-step exploration (no existing steps found)")
        # Load initial beliefs from file if specified
        if (beliefs_path := config.eval.get("beliefs_path", None)) is not None:
            h = Path(beliefs_path).read_text()
            evolve_logger.info(f"Loaded initial beliefs from: {beliefs_path}")
        else:
            h = ""

        # Load initial perception from file if specified
        if (perception_path := config.eval.get("perception_path", None)) is not None:
            p = Path(perception_path).read_text()
            evolve_logger.info(f"Loaded initial perception from: {perception_path}")
        else:
            p = ""
    # Hypotheses start empty each step; Phase A.2 generates them fresh
    hypotheses = []

    config.eval.num_episodes = explore_config.rollouts_per_step

    # Fetch default knowledge once
    default_knowledge = get_default_knowledge(config)
    evolve_logger.info(f"Default knowledge extraction complete (length: {len(default_knowledge)})")

    evolve_logger.info(f"Starting 2-step exploration with {explore_config.num_steps} steps")
    evolve_logger.info(f"Rollouts per step: {explore_config.rollouts_per_step}")
    evolve_logger.info(f"Hypotheses per step: {explore_config.num_hypotheses}")
    evolve_logger.info(f"Baseline rollouts: {explore_config.num_baseline_rollouts}")

    cumulative_cost = 0.0

    for step in range(start_step, explore_config.num_steps + 1):
        evolve_logger.info(f"\n{'='*80}")
        evolve_logger.info(f"2-STEP EXPLORATION STEP {step}/{explore_config.num_steps}")
        evolve_logger.info(f"{'='*80}")

        step_output_dir = Path(output_dir) / f"step_{step}"
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Save current state inputs
        (step_output_dir / "input_beliefs.txt").write_text(h)
        (step_output_dir / "input_perception.txt").write_text(p)

        # ================================================================
        # PHASE A: Baseline (no hypotheses)
        # ================================================================
        evolve_logger.info(f"--- Phase A: Baseline (no hypotheses) ---")

        baseline_dir = step_output_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_rollout_dir = baseline_dir / "rollouts"
        baseline_rollout_dir.mkdir(parents=True, exist_ok=True)

        # Phase A.1: Baseline rollouts (no hypotheses)
        with step_logging(baseline_dir) as step_log_file:
            logging.info(f"Step {step} Phase A: Baseline rollout logs")
            logging.info(f"Current beliefs:\n{h if h else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")
            logging.info("=== Phase A.1: Baseline Rollouts (no hypotheses) ===")

            baseline_rollout_results = run_explore_rollouts(
                base_beliefs=h,
                perception=p,
                hypotheses=[],  # No hypotheses - just play the game
                config=config,
                original_cwd=original_cwd,
                output_dir=str(baseline_rollout_dir),
                num_baseline_rollouts=explore_config.num_baseline_rollouts,
            )

        # Save baseline rollout stats
        with open(baseline_dir / "rollout_stats.json", "w") as f:
            json.dump(baseline_rollout_results, f, indent=4, default=str)

        _log_rollout_stats(baseline_rollout_results, "Baseline")

        # Phase A.2: Baseline improve (generates hypotheses for Phase B)
        with improve_logging(baseline_dir) as improve_log_file:
            logging.info(f"Step {step} Phase A: Baseline improve logs")
            logging.info("=== Phase A.2: Baseline Improve (generates hypotheses) ===")
            logging.info(f"Current beliefs:\n{h if h else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")

            new_h, new_p, new_hypotheses, baseline_improve_cost = improve_step(
                config=config,
                base_beliefs=h,
                perception=p,
                output_dir=str(baseline_rollout_dir),
                previous_hypotheses=[],  # No hypotheses were tested in baseline
                default_knowledge=default_knowledge,
                num_hypotheses=explore_config.num_hypotheses,
                rollout_results=baseline_rollout_results,
            )

        # Update state from baseline phase
        h = new_h
        p = new_p
        hypotheses = new_hypotheses

        # Save baseline phase outputs
        (baseline_dir / "beliefs.txt").write_text(h)
        (baseline_dir / "perception.py").write_text(p)
        with open(baseline_dir / "hypotheses.json", "w") as f:
            json.dump(hypotheses, f, indent=4)

        baseline_rollout_cost = _compute_rollout_cost(baseline_rollout_results)
        baseline_total_cost = baseline_rollout_cost + baseline_improve_cost
        evolve_logger.info(
            f"Phase A costs: rollout=${baseline_rollout_cost:.6f}, "
            f"improve=${baseline_improve_cost:.6f}, total=${baseline_total_cost:.6f}"
        )
        evolve_logger.info(f"Phase A updated beliefs:\n{h}")
        evolve_logger.info(f"Phase A generated {len(hypotheses)} hypotheses for Phase B:")
        for i, hyp in enumerate(hypotheses):
            evolve_logger.info(f"  Hypothesis {i+1}: {hyp}")

        # ================================================================
        # PHASE B: Explore (with hypotheses from Phase A)
        # ================================================================
        evolve_logger.info(f"--- Phase B: Explore (with hypotheses from Phase A) ---")
        evolve_logger.info(f"Hypotheses to test: {len(hypotheses)}")
        if hypotheses:
            for i, hyp in enumerate(hypotheses):
                evolve_logger.info(f"  Hypothesis {i+1}: {hyp}")
        else:
            evolve_logger.info("  (no hypotheses generated - running baseline rollouts for explore phase)")

        explore_dir = step_output_dir / "explore"
        explore_dir.mkdir(parents=True, exist_ok=True)
        explore_rollout_dir = explore_dir / "rollouts"
        explore_rollout_dir.mkdir(parents=True, exist_ok=True)

        # Phase B.1: Explore rollouts (with hypotheses)
        with step_logging(explore_dir) as step_log_file:
            logging.info(f"Step {step} Phase B: Explore rollout logs")
            logging.info(f"Current beliefs:\n{h if h else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")
            logging.info("=== Phase B.1: Explore Rollouts (with hypotheses) ===")

            explore_rollout_results = run_explore_rollouts(
                base_beliefs=h,
                perception=p,
                hypotheses=hypotheses,
                config=config,
                original_cwd=original_cwd,
                output_dir=str(explore_rollout_dir),
                num_baseline_rollouts=explore_config.num_baseline_rollouts,
            )

        # Save explore rollout stats
        with open(explore_dir / "rollout_stats.json", "w") as f:
            json.dump(explore_rollout_results, f, indent=4, default=str)

        _log_rollout_stats(explore_rollout_results, "Explore")

        # Phase B.2: Explore improve (evaluates hypotheses, does NOT generate new ones)
        with improve_logging(explore_dir) as improve_log_file:
            logging.info(f"Step {step} Phase B: Explore improve logs")
            logging.info("=== Phase B.2: Explore Improve (evaluate hypotheses, no generation) ===")
            logging.info(f"Current beliefs:\n{h if h else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")

            new_h, new_p, explore_improve_cost = improve_step_no_hypotheses(
                config=config,
                base_beliefs=h,
                perception=p,
                output_dir=str(explore_rollout_dir),
                previous_hypotheses=hypotheses,
                default_knowledge=default_knowledge,
                rollout_results=explore_rollout_results,
            )

        # Update state from explore phase (no new hypotheses - Phase A generates them)
        h = new_h
        p = new_p

        # Save explore phase outputs
        (explore_dir / "beliefs.txt").write_text(h)
        (explore_dir / "perception.py").write_text(p)

        explore_rollout_cost = _compute_rollout_cost(explore_rollout_results)
        explore_total_cost = explore_rollout_cost + explore_improve_cost
        evolve_logger.info(
            f"Phase B costs: rollout=${explore_rollout_cost:.6f}, "
            f"improve=${explore_improve_cost:.6f}, total=${explore_total_cost:.6f}"
        )

        total_step_cost = baseline_total_cost + explore_total_cost
        cumulative_cost += total_step_cost
        evolve_logger.info(f"Step {step} total cost: ${total_step_cost:.6f}")
        evolve_logger.info(f"Cumulative cost after step {step}: ${cumulative_cost:.6f}")

        # Log summary
        evolve_logger.info(f"Step {step} completed.")
        evolve_logger.info(f"Updated beliefs:\n{h}")


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
        run_name = f"{timestamp}_{config.agent.type}_{config.client.model_id.replace('/', '_')}_explore2step"
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

    # Get exploration config with defaults
    num_hypotheses = config.eval.evolve.get("num_hypotheses", 1)
    num_baseline_rollouts = config.eval.evolve.get("num_baseline_rollouts", 3)

    match config.eval.mode:
        case "eval":
            # Simple eval mode - just run one step
            from rollout import one_step_wrap
            one_step_wrap(config=config, original_cwd=original_cwd, output_dir=output_dir)

        case "explore":
            ec = ExploreConfig(
                num_steps=config.eval.evolve.num_steps,
                rollouts_per_step=config.eval.evolve.rollouts_per_step,
                num_hypotheses=num_hypotheses,
                num_baseline_rollouts=num_baseline_rollouts,
            )
            online_explore_2step(
                explore_config=ec,
                config=config,
                original_cwd=original_cwd,
                output_dir=output_dir,
            )

        case _:
            evolve_logger.error(f"Unsupported mode: {config.eval.mode}. explore_2step_evolve.py supports 'eval' and 'explore' modes.")
            raise ValueError(f"Unsupported mode: {config.eval.mode}")


if __name__ == "__main__":
    main()
