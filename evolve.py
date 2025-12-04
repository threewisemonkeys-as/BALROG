import asyncio
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import hydra
import litellm
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from balrog.agents import AgentFactory
from balrog.evaluator import EvaluatorManager
from balrog.utils import collect_and_summarize_results, print_summary_table, setup_environment
from llm_utils import build_llm_input, extract_llm_response_text, extract_xml_kv, validate_response_fields


def summarise_results(results: dict):
    summary = {}
    for env_name, env_results in results.items():
        num_attempts = len(env_results)
        summary[env_name] = {
            "avg_prog": sum([r['progression'] for r in env_results]) / num_attempts,
            "avg_steps": sum([r['num_steps'] for r in env_results]) / num_attempts,
            "avg_cost": sum([r['total_cost'] for r in env_results]) / num_attempts,
        }
    
    return summary


def one_step(
    instruction: str,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    config.instruction_prompt = instruction
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
    if (instruction_path := config.get("instruction_path", None)) is None:
        logging.info(f"Path to instructions not specified as instruction_path arg, considering empty instruction")
        instruction = ""
    else:
        instruction = Path(instruction_path).read_text()

    logging.info(f"Using following as instruction -\n{instruction}")

    summary = one_step(
        instruction=instruction,
        config=config,
        original_cwd=original_cwd,
        output_dir=output_dir,
    )

    print(summary)


async def get_episode_summary_async(
    config: DictConfig,
    instruction: str,
    trajectory_path: str
) -> str:
    """Get a summary of an episode trajectory using LLM (async).

    Args:
        config: Configuration containing model information
        instruction: Current beliefs/instructions about the game
        trajectory_path: Path to the trajectory file

    Returns:
        Summary text extracted from LLM response
    """
    traj_text = Path(trajectory_path).read_text()

    prompt = f"""We are playing a game and trying to figure out how it works.
Currently we have the following list of beliefs about the game -

{instruction}

We have played the game using these beliefs resulting in the following trajectory -

{traj_text}

Provide a short summary of the trajectory, highlighting key decisions made and mistakes made if any.
Format your response in XML style as -
<think>
Think about the trajectory followed when playing the game.
</think>
<summary>
Summary as requested
</summary>
"""

    # Build input for LLM
    input_data = build_llm_input(prompt)

    # Call LLM asynchronously
    logging.info(f"Calling LLM to summarize trajectory: {trajectory_path}")
    response = await asyncio.to_thread(
        litellm.responses,
        model=f"{config.client.client_name}/{config.client.model_id}",
        input=input_data,
        num_retries=5,
    )

    # Extract response text
    response_text = extract_llm_response_text(response)

    # Extract summary from XML
    response_dict = extract_xml_kv(response_text, ["summary"])
    

    if not validate_response_fields(response_dict, response_text, ["summary"]):
        return ""
    else:
        return response_dict["summary"]


def get_episode_summary(
    config: DictConfig,
    instruction: str,
    trajectory_path: str
) -> str:
    """Get a summary of an episode trajectory using LLM (sync wrapper).

    Args:
        config: Configuration containing model information
        instruction: Current beliefs/instructions about the game
        trajectory_path: Path to the trajectory file

    Returns:
        Summary text extracted from LLM response
    """
    return asyncio.run(get_episode_summary_async(config, instruction, trajectory_path))

def improve(
    config: DictConfig,
    instruction: str,
    output_dir: str,
) -> str:
    """Improve beliefs based on episode trajectories using LLM.

    Args:
        config: Configuration containing model information
        instruction: Current beliefs/instructions about the game
        output_dir: Directory containing episode trajectory files

    Returns:
        Updated beliefs as a string
    """

    async def get_all_summaries():
        """Get all episode summaries in parallel."""
        episode_paths = list(Path(output_dir).rglob("*.csv"))
        logging.info(f"Processing {len(episode_paths)} episodes in parallel")

        tasks = [
            get_episode_summary_async(config, instruction, str(episode_path))
            for episode_path in episode_paths
        ]
        return await asyncio.gather(*tasks)

    # Get summaries for all episodes in parallel
    ep_summaries = asyncio.run(get_all_summaries())

    # Combine all summaries
    ep_summaries_str = ""
    for ep_sum in ep_summaries:
        ep_summaries_str += f"Episode Summary:\n{ep_sum.strip()}\n\n"

    # Build prompt for belief update
    prompt = f"""We are playing a game and trying to figure out how it works.
Currently we have the following list of beliefs about the game -

{instruction}

We have played the game using these beliefs with the following results -

{ep_summaries_str}

Think about how we need to update our beliefs given this information.
Only make updates to the set of beliefs based on the results provided above, do not assume anything about the game.
Format your response in XML style as -
<think>
think about how we need to update our beliefs
</think>
<beliefs>
- updated belief
- updated belief
...
</beliefs>"""

    # Build input for LLM
    input_data = build_llm_input(prompt)

    # Call LLM
    logging.info(f"Calling LLM to generate updated beliefs with prompt -\n{prompt}")
    response = litellm.responses(
        model=f"{config.client.client_name}/{config.client.model_id}",
        input=input_data,
        num_retries=5,
    )

    # Extract response text
    response_text = extract_llm_response_text(response)

    # Extract beliefs from XML
    response_dict = extract_xml_kv(response_text, ["beliefs"])
    validate_response_fields(response_dict, response_text, ["beliefs"])

    updated_beliefs = response_dict["beliefs"].strip()
    logging.info(f"Updated beliefs:\n{updated_beliefs}")

    return updated_beliefs



@dataclass
class EvolveConfig:
    num_steps: int
    rollouts_per_step: int


def find_last_completed_step(output_dir: str) -> tuple[int, str]:
    """Find the last completed step with a beliefs.txt file.

    Args:
        output_dir: Output directory containing step folders

    Returns:
        Tuple of (last_step_number, beliefs_content). Returns (0, "") if no steps found.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0, ""

    # Find all step directories
    step_dirs = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            try:
                step_num = int(item.name.split("_")[1])
                beliefs_file = item / "beliefs.txt"
                if beliefs_file.exists():
                    step_dirs.append((step_num, beliefs_file))
            except (ValueError, IndexError):
                continue

    if not step_dirs:
        return 0, ""

    # Sort by step number and get the last one
    step_dirs.sort(key=lambda x: x[0])
    last_step_num, last_beliefs_file = step_dirs[-1]
    beliefs_content = last_beliefs_file.read_text()

    logging.info(f"Found last completed step: {last_step_num}")
    logging.info(f"Resuming with beliefs from: {last_beliefs_file}")

    return last_step_num, beliefs_content


def online_evolve(
    evolve_config: EvolveConfig,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    """Run online evolution loop.

    Args:
        evolve_config: Evolution configuration
        config: BALROG configuration
        original_cwd: Original working directory
        output_dir: Output directory for results
    """
    # Check for existing progress and resume if available
    last_step, h = find_last_completed_step(output_dir)
    start_step = last_step + 1

    if last_step > 0:
        logging.info(f"Resuming from step {start_step} (found {last_step} completed steps)")
        logging.info(f"Loaded beliefs:\n{h}")
    else:
        logging.info("Starting fresh evolution (no existing steps found)")
        h = ""

    config.num_episodes = evolve_config.rollouts_per_step

    logging.info(f"Starting online evolution with {evolve_config.num_steps} steps")
    logging.info(f"Rollouts per step: {evolve_config.rollouts_per_step}")

    for step in range(start_step, evolve_config.num_steps + 1):
        logging.info(f"\n{'='*80}")
        logging.info(f"STEP {step}/{evolve_config.num_steps}")
        logging.info(f"{'='*80}")

        step_output_dir = Path(output_dir) / f"step_{step}"
        step_output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Current beliefs:\n{h if h else '(empty)'}")

        # Run rollouts with current beliefs
        step_summary = one_step(
            instruction=h,
            config=config,
            original_cwd=original_cwd,
            output_dir=str(step_output_dir),
        )

        logging.info(f"Step {step} summary: {step_summary}")

        # Improve beliefs based on results
        h = improve(config, h, str(step_output_dir))

        # Save updated beliefs
        beliefs_file = step_output_dir / "beliefs.txt"
        beliefs_file.write_text(h)
        logging.info(f"Saved updated beliefs to {beliefs_file}")
            



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
        run_name = f"{timestamp}_{config.agent.type}_{config.client.model_id.replace('/', '_')}"
        output_dir = os.path.join(config.eval.output_dir, run_name)

        # Create the directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_filename = os.path.join(output_dir, "eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename)],
        force=True,
    )

    # Create an EvaluatorManager and run evaluation
    # with redirect_to_file(log_filename):
        # one_step(config, original_cwd, output_dir)

    # one_step_wrap(config=config, original_cwd=original_cwd, output_dir=output_dir)

    ec = EvolveConfig(
        num_steps=20,
        rollouts_per_step=1,
    )
    online_evolve(
        evolve_config=ec,
        config=config,
        original_cwd=original_cwd,
        output_dir=output_dir,
    )



if __name__ == "__main__":
    main()
