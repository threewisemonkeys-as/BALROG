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


def validate_perception_code(code: str) -> tuple[bool, str | None]:
    """Validate perception code by attempting to compile and execute it.

    Args:
        code: Python code string containing the perceive function

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    if not code or not code.strip():
        return True, None  # Empty code is valid (no perception)

    # First, try to compile the code to catch syntax errors
    try:
        compile(code, "<perception_module>", "exec")
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"

    # Try to execute and verify the perceive function exists and is callable
    try:
        namespace = {}
        exec(code, namespace)
        if "perceive" not in namespace:
            return False, "No 'perceive' function found in the code"
        if not callable(namespace["perceive"]):
            return False, "'perceive' is not a callable function"

        # Test with a sample input to catch runtime errors in basic execution
        test_input = "message: test\ncursor: x=0, y=0\nmap:\n...\n"
        try:
            result = namespace["perceive"](test_input)
            if not isinstance(result, str):
                return False, f"'perceive' function must return a string, got {type(result).__name__}"
        except Exception as e:
            return False, f"Runtime error when testing perceive function: {e}"

    except Exception as e:
        return False, f"Failed to execute perception code: {e}"

    return True, None

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


@contextmanager
def improve_logging(step_output_dir: Path):
    """Context manager to redirect all logging to an improve-specific log file.

    During the context, all log messages go to step_output_dir/improve.log.
    After exiting, logging returns to normal (eval.log).
    This is used for logging LLM prompts and outputs during the improve phase.
    """
    improve_log_file = step_output_dir / "improve.log"

    # Create a handler for the improve log
    improve_handler = logging.FileHandler(improve_log_file)
    improve_handler.setLevel(logging.INFO)
    improve_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Get the root logger and save its current handlers
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()

    # Replace handlers with improve handler
    root_logger.handlers = [improve_handler]

    try:
        yield improve_log_file
    finally:
        # Restore original handlers
        improve_handler.close()
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


def improve_step(
    config: DictConfig,
    base_beliefs: str,
    perception: str,
    output_dir: str,
    previous_experiments: list[str],
    default_knowledge: str,
    num_experiments: int = 1,
    rollout_results: dict[str, dict] | None = None,
) -> tuple[str, str, list[str], float]:
    """Improve step: Evaluate, Update Beliefs/Perception, Generate New Experiments.

    Args:
        config: Configuration containing model information
        base_beliefs: Current beliefs/instructions
        perception: Current perception module
        output_dir: Directory containing rollout results
        previous_experiments: List of experiments tested in the last step
        default_knowledge: Default knowledge string to include in prompt
        num_experiments: Number of new experiments to generate
        rollout_results: Results from run_explore_rollouts, including any errors

    Returns:
        Tuple of (updated_beliefs, updated_perception, new_hypotheses, total_improve_cost)
    """

    async def get_all_summaries():
        """Get all episode summaries in parallel."""
        # First, identify all episode paths and map them to their experiment (if any)
        episode_tasks = []

        for episode_path in Path(output_dir).rglob("*.csv"):
            try:
                # Get path relative to output_dir to determine run name
                rel_path = episode_path.relative_to(output_dir)
                run_name = rel_path.parts[0] # e.g., baseline_0, experiment_1

                experiment_text = None
                if run_name.startswith("experiment_"):
                    try:
                        idx = int(run_name.split("_")[1])
                        if 0 <= idx < len(previous_experiments):
                            experiment_text = previous_experiments[idx]
                    except (ValueError, IndexError):
                        logging.warning(f"Could not map run {run_name} to an experiment")
                
                # Create task for summary generation
                episode_tasks.append(
                    get_episode_summary_async(
                        config, 
                        base_beliefs, 
                        perception, 
                        str(episode_path), 
                        experiment=experiment_text
                    )
                )
            except ValueError:
                logging.warning(f"Could not determine run name for {episode_path}")
                continue
                
        logging.info(f"Generating summaries for {len(episode_tasks)} episodes in parallel")
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

We have collected new experience.
{evidence_section}

Your task is to:
1. Analyze the results. If experiments were tested, determine if they were confirmed or refuted.
2. Update our beliefs about the game based on confirmed knowledge.
3. Update the perception module to make sure it is correct and that it extracts better features from the direct game observation.
4. Generate {num_experiments} NEW experiments to test in the next step.
   - Experiments should be specific, actionable strategies or mechanics to test.
   - They should help us achieve the main goal.

It is important that you keep both the beliefs and the output of the perception module as simple as possible.

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

Format your response in XML style as:
<think>
Analyze results, evaluate experiments, determine belief updates, design perception improvements, and brainstorm new experiments.
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
<new_hypotheses>
HYPOTHESIS 1: [First hypothesis to test]

HYPOTHESIS 2: [Second hypothesis to test]
...
(Generate exactly {num_hypotheses} hypotheses)
</new_hypotheses>
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
    new_hypotheses = []
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

        logging.info(f"Improve step prompt (attempt {attempt + 1}/{max_retries}):\n{prompt}")

        # Build input for LLM
        input_data = build_llm_input(prompt)

        # Call LLM
        logging.info(f"Calling LLM for improve step (attempt {attempt + 1}/{max_retries})")
        response = litellm.responses(
            model=model_name,
            input=input_data,
            num_retries=5,
        )
        improve_call_cost += _get_response_cost(response, config.client.model_id)

        # Extract response text
        response_text = extract_llm_response_text(response)
        logging.info(f"Improve step LLM response (attempt {attempt + 1}/{max_retries}):\n{response_text}")

        # Extract fields
        response_dict = extract_xml_kv(response_text, ["updated_beliefs", "perception", "new_hypotheses"])
        validate_response_fields(response_dict, response_text, ["updated_beliefs", "perception", "new_hypotheses"])

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
                # On final attempt, keep the previous working perception
                logging.error(f"All {max_retries} attempts to generate valid perception code failed. Keeping previous perception.")
                updated_perception = perception

        # Process hypotheses (do this on every attempt so we have them even if perception fails)
        if "new_hypotheses" in response_dict:
            hypotheses_text = response_dict["new_hypotheses"].strip()
            import re
            hypothesis_pattern = r'HYPOTHESIS\s*\d+\s*:\s*(.+?)(?=HYPOTHESIS\s*\d+\s*:|$)'
            matches = re.findall(hypothesis_pattern, hypotheses_text, re.DOTALL | re.IGNORECASE)
            new_hypotheses = [m.strip() for m in matches if m.strip()]

            # Fallback split
            if not new_hypotheses:
                new_hypotheses = [h.strip() for h in hypotheses_text.split('\n\n') if h.strip()]

    # Process hypotheses from final response if not already processed
    if not new_hypotheses and "new_hypotheses" in response_dict:
        hypotheses_text = response_dict["new_hypotheses"].strip()
        import re
        hypothesis_pattern = r'HYPOTHESIS\s*\d+\s*:\s*(.+?)(?=HYPOTHESIS\s*\d+\s*:|$)'
        matches = re.findall(hypothesis_pattern, hypotheses_text, re.DOTALL | re.IGNORECASE)
        new_hypotheses = [m.strip() for m in matches if m.strip()]

        # Fallback split
        if not new_hypotheses:
            new_hypotheses = [h.strip() for h in hypotheses_text.split('\n\n') if h.strip()]

    # Limit to requested num
    new_hypotheses = new_hypotheses[:num_hypotheses]

    logging.info(f"Updated beliefs:\n{updated_beliefs}")
    logging.info(f"Updated perception:\n{updated_perception}")
    logging.info(f"Generated {len(new_hypotheses)} new hypotheses")

    total_improve_cost = summary_cost + improve_call_cost
    return updated_beliefs, updated_perception, new_hypotheses, total_improve_cost


@dataclass
class ExploreConfig:
    num_steps: int
    rollouts_per_step: int
    num_hypotheses: int
    num_baseline_rollouts: int


def get_default_knowledge(config: DictConfig) -> str:
    """Get the default instructions/knowledge for the environment.

    Args:
        config: BALROG configuration

    Returns:
        String containing default instructions (actions, goal, etc.)
    """
    env_name = config.envs.names.split("-")[0]
    # Get the first task for this environment
    tasks = config.tasks[f"{env_name}_tasks"]
    if not tasks:
        return ""
    
    task = tasks[0]
    logging.info(f"Extracting default knowledge for env: {env_name}, task: {task}")
    
    try:
        env = make_env(env_name, task, config)
        instruction_prompt = get_loaded_instruction_prompt(env, load="", task=task)
        env.close()
        return instruction_prompt
    except Exception as e:
        logging.warning(f"Failed to extract default knowledge: {e}")
        return ""


def find_last_completed_step(output_dir: str) -> tuple[int, str, str, list[str]]:
    """Find the last completed step with beliefs.txt, perception.py, and hypotheses.json files.

    Args:
        output_dir: Output directory containing step folders

    Returns:
        Tuple of (last_step_number, beliefs_content, perception_content, hypotheses_list).
        Returns (0, "", "", []) if no steps found.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0, "", "", []

    # Find all step directories
    step_dirs = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            try:
                step_num = int(item.name.split("_")[1])
                beliefs_file = item / "beliefs.txt"
                perception_file = item / "perception.py"
                hypotheses_file = item / "hypotheses.json"
                if beliefs_file.exists() and perception_file.exists():
                    step_dirs.append((step_num, beliefs_file, perception_file, hypotheses_file))
            except (ValueError, IndexError):
                continue

    if not step_dirs:
        return 0, "", "", []

    # Sort by step number and get the last one
    step_dirs.sort(key=lambda x: x[0])
    last_step_num, last_beliefs_file, last_perception_file, last_hypotheses_file = step_dirs[-1]
    beliefs_content = last_beliefs_file.read_text()
    perception_content = last_perception_file.read_text()

    hypotheses = []
    if last_hypotheses_file.exists():
        try:
            hypotheses = json.loads(last_hypotheses_file.read_text())
        except json.JSONDecodeError:
            hypotheses = []

    evolve_logger.info(f"Found last completed step: {last_step_num}")
    evolve_logger.info(f"Resuming with beliefs from: {last_beliefs_file}")
    evolve_logger.info(f"Resuming with perception from: {last_perception_file}")

    return last_step_num, beliefs_content, perception_content, hypotheses


def online_explore(
    explore_config: ExploreConfig,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    """Run online exploration loop with unified Explore -> Improve steps.

    Args:
        explore_config: Exploration configuration
        config: BALROG configuration
        original_cwd: Original working directory
        output_dir: Output directory for results
    """
    evolve_logger.info(f"Running exploration with hypothesis generation")

    # Check for existing progress and resume if available
    last_step, h, p, hypotheses = find_last_completed_step(output_dir)
    start_step = last_step + 1

    if last_step > 0:
        evolve_logger.info(f"Resuming from step {start_step} (found {last_step} completed steps)")
        evolve_logger.info(f"Resuming with {len(hypotheses)} hypotheses from previous step")
    else:
        evolve_logger.info("Starting fresh exploration (no existing steps found)")
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
        hypotheses = []  # Start with no hypotheses -> will trigger baseline rollouts

    config.eval.num_episodes = explore_config.rollouts_per_step

    # Fetch default knowledge once
    default_knowledge = get_default_knowledge(config)
    evolve_logger.info(f"Default knowledge extraction complete (length: {len(default_knowledge)})")

    evolve_logger.info(f"Starting exploration with {explore_config.num_steps} steps")
    evolve_logger.info(f"Rollouts per step: {explore_config.rollouts_per_step}")
    evolve_logger.info(f"Hypotheses per step: {explore_config.num_hypotheses}")
    evolve_logger.info(f"Baseline rollouts: {explore_config.num_baseline_rollouts}")

    cumulative_cost = 0.0

    for step in range(start_step, explore_config.num_steps + 1):
        evolve_logger.info(f"\n{'='*80}")
        evolve_logger.info(f"EXPLORATION STEP {step}/{explore_config.num_steps}")
        evolve_logger.info(f"Hypotheses to test: {len(hypotheses)}")
        if hypotheses:
            for i, hyp in enumerate(hypotheses):
                evolve_logger.info(f"  Hypothesis {i+1}: {hyp}")
        else:
            evolve_logger.info(f"  (no hypotheses - running baseline rollouts)")
        evolve_logger.info(f"{'='*80}")

        step_output_dir = Path(output_dir) / f"step_{step}"
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Save current state inputs (outside logging context)
        (step_output_dir / "input_beliefs.txt").write_text(h)
        (step_output_dir / "input_perception.txt").write_text(p)
        with open(step_output_dir / "input_hypotheses.json", "w") as f:
            json.dump(hypotheses, f, indent=4)

        rollout_dir = step_output_dir / "rollouts"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Explore (Rollouts) - with step-specific logging for agent steps
        with step_logging(step_output_dir) as step_log_file:
            logging.info(f"Step {step} agent rollout logs")
            logging.info(f"Current beliefs:\n{h if h else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")

            logging.info("=== Phase 1: Explore (Rollouts) ===")

            rollout_results = run_explore_rollouts(
                base_beliefs=h,
                perception=p,
                hypotheses=hypotheses,
                config=config,
                original_cwd=original_cwd,
                output_dir=str(rollout_dir),
                num_baseline_rollouts=explore_config.num_baseline_rollouts,
            )

        # Save detailed rollout stats (outside logging context)
        with open(step_output_dir / "rollout_stats.json", "w") as f:
            json.dump(rollout_results, f, indent=4, default=str)

        # Calculate rollout success statistics and log to eval.log
        total_rollouts = len(rollout_results)
        successful_rollouts = 0
        partial_rollouts = 0
        failed_rollouts = 0
        error_rollouts = 0

        for run_name, result in rollout_results.items():
            if "error" in result:
                error_rollouts += 1
                continue
            # Check summary for each environment
            summary = result.get("summary", {})
            for env_name, env_stats in summary.items():
                if env_stats.get("num_perfect", 0) > 0:
                    successful_rollouts += 1
                elif env_stats.get("avg_prog", 0) > 0:
                    partial_rollouts += 1
                else:
                    failed_rollouts += 1

        evolve_logger.info(f"Rollout results: {total_rollouts} total, {successful_rollouts} successful (100%), "
                          f"{partial_rollouts} partial progress, {failed_rollouts} failed (0%), {error_rollouts} errors")

        # Phase 2: Improve (Analyze -> Update -> Generate Hypotheses) - with improve-specific logging
        with improve_logging(step_output_dir) as improve_log_file:
            logging.info(f"Step {step} improve phase logs")
            logging.info("=== Phase 2: Improve (Update & Generate) ===")
            logging.info(f"Current beliefs:\n{h if h else '(empty)'}")
            logging.info(f"Current perception:\n{p if p else '(empty)'}")

            new_h, new_p, new_hypotheses, improve_cost = improve_step(
                config=config,
                base_beliefs=h,
                perception=p,
                output_dir=str(rollout_dir),
                previous_hypotheses=hypotheses,
                default_knowledge=default_knowledge,
                num_hypotheses=explore_config.num_hypotheses,
                rollout_results=rollout_results,
            )

        # Update state for next step
        h = new_h
        p = new_p
        hypotheses = new_hypotheses

        # Save updated state (outside logging context)
        (step_output_dir / "beliefs.txt").write_text(h)
        (step_output_dir / "perception.py").write_text(p)
        with open(step_output_dir / "hypotheses.json", "w") as f:
            json.dump(hypotheses, f, indent=4)

        # Calculate and log costs
        rollout_cost = 0.0
        for _, result in rollout_results.items():
            if "summary" in result:
                for env_stats in result["summary"].values():
                    rollout_cost += env_stats.get("total_cost", 0)
        total_step_cost = rollout_cost + improve_cost
        cumulative_cost += total_step_cost
        evolve_logger.info(f"Step {step} costs: rollout=${rollout_cost:.6f}, improve=${improve_cost:.6f}, total=${total_step_cost:.6f}")
        evolve_logger.info(f"Cumulative cost after step {step}: ${cumulative_cost:.6f}")

        # Log summary to eval.log
        evolve_logger.info(f"Step {step} completed.")
        evolve_logger.info(f"Updated beliefs:\n{h}")
        evolve_logger.info(f"Generated {len(hypotheses)} new hypotheses for next step:")
        for i, hyp in enumerate(hypotheses):
            evolve_logger.info(f"  Hypothesis {i+1}: {hyp}")


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
        run_name = f"{timestamp}_{config.agent.type}_{config.client.model_id.replace('/', '_')}_explore"
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
            online_explore(
                explore_config=ec,
                config=config,
                original_cwd=original_cwd,
                output_dir=output_dir,
            )

        case _:
            evolve_logger.error(f"Unsupported mode: {config.eval.mode}. explore_eval.py supports 'eval' and 'explore' modes.")
            raise ValueError(f"Unsupported mode: {config.eval.mode}")


if __name__ == "__main__":
    main()
