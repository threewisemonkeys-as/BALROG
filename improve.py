import asyncio
import json
import logging
import re
from pathlib import Path

import litellm
from omegaconf import DictConfig

from balrog.pricing import calculate_cost
from llm_utils import build_llm_input, extract_llm_response_text, extract_xml_kv, validate_response_fields


# Map NLE end_status codes to human-readable descriptions
# Source: nle/env/base.py:152-166 and nle/env/tasks.py:102-106
END_REASON_MAP = {
    -1: "ABORTED - Episode truncated (max steps reached)",
    0: "RUNNING - Episode ended while still in progress",
    1: "DEATH - Agent died during the episode",
    2: "TASK_SUCCESSFUL - Agent completed the task goal",
}


def _get_response_cost(response, model_id: str) -> float:
    """Extract cost from a litellm response object using BALROG pricing."""
    try:
        usage = response.usage
        if usage is None:
            return 0.0
        input_tokens = getattr(usage, 'input_tokens', 0) or 0
        output_tokens = getattr(usage, 'output_tokens', 0) or 0
        return calculate_cost(model_id, input_tokens, output_tokens)
    except Exception:
        return 0.0


def get_episode_outcome_header(trajectory_path: str) -> str:
    """Read episode metadata from JSON file and create a clear outcome header.

    Args:
        trajectory_path: Path to the trajectory CSV file

    Returns:
        Formatted string describing the episode outcome
    """
    # Find corresponding JSON file (same name but .json extension)
    csv_path = Path(trajectory_path)
    json_path = csv_path.with_suffix(".json")

    if not json_path.exists():
        return "=== EPISODE OUTCOME: Unknown (metadata file not found) ===\n"

    try:
        with open(json_path) as f:
            metadata = json.load(f)

        progression = metadata.get("progression", 0.0)
        num_steps = metadata.get("num_steps", "unknown")
        end_reason_code = metadata.get("end_reason", None)
        episode_return = metadata.get("episode_return", 0.0)
        task = metadata.get("task", "unknown")

        # Convert end_reason code to human-readable string
        if isinstance(end_reason_code, int):
            end_reason_str = END_REASON_MAP.get(end_reason_code, f"Unknown (code={end_reason_code})")
            # Include raw code for transparency
            end_reason_str = f"{end_reason_str} (code={end_reason_code})"
        elif isinstance(end_reason_code, str):
            end_reason_str = end_reason_code
        else:
            end_reason_str = f"Unknown (raw={end_reason_code})"

        # Determine success/failure status
        if progression >= 1.0:
            outcome_status = "SUCCESS - Goal achieved!"
        elif progression > 0:
            outcome_status = f"PARTIAL SUCCESS ({progression*100:.0f}% progress toward goal)"
        else:
            outcome_status = "FAILURE - No progress toward goal (0%)"

        # Provide goal context based on task name
        if "Quest" in task or "Staircase" in task:
            goal_reminder = "GOAL: Find and reach the stairs down (>) to descend to the next level."
        elif "Oracle" in task:
            goal_reminder = "GOAL: Find and reach the Oracle."
        elif "Gold" in task:
            goal_reminder = "GOAL: Collect as much gold as possible."
        elif "Eat" in task:
            goal_reminder = "GOAL: Find and consume food to stay alive."
        elif "Scout" in task:
            goal_reminder = "GOAL: Explore and discover as much of the map as possible."
        else:
            goal_reminder = "GOAL: Complete the task objective."

        header = f"""=== EPISODE OUTCOME ===
Task: {task}
{goal_reminder}
Status: {outcome_status}
End Reason: {end_reason_str}
Steps Taken: {num_steps}
Progression: {progression*100:.1f}%
Episode Return: {episode_return:.2f}
===========================
"""
        return header

    except (json.JSONDecodeError, KeyError) as e:
        logging.warning(f"Failed to parse episode metadata from {json_path}: {e}")
        return "=== EPISODE OUTCOME: Unknown (failed to parse metadata) ===\n"


def trim_to_model_context_lim(text: str, model_name: str, buffer: int = 0, prefix: bool = True) -> str:
    if "qwen" in model_name.lower():
        return text[-100000:]
    else:
        return text[-350000:]


async def _get_instructions_perception_summary_async(
    config: DictConfig,
    instruction: str,
    perception: str,
    outcome_header: str,
    traj_text: str,
    trajectory_path: str,
) -> str:
    """Get a summary focused on instructions and perception evaluation (async).

    Args:
        config: Configuration containing model information
        instruction: Current beliefs/instructions about the game
        perception: Current perception module code
        outcome_header: Formatted episode outcome header
        traj_text: Trajectory text content
        trajectory_path: Path to the trajectory file (for logging)

    Returns:
        Summary text focused on instructions and perception
    """
    prompt = f"""We are playing a game and trying to figure out how it works.
Currently we have the following list of beliefs about the game -

{instruction if instruction else "(empty - no beliefs)"}

We are also using the following perception module to process game observations -

{perception if perception else "(empty - no perception module)"}

We have played the game using these beliefs and perception.  
Here is the episode outcome and trajectory:

{outcome_header}
{traj_text}

Your summary should contain analysis of the behaviour of the perception module:
- The perception module is provided everything inside the Direct Game Observation as an input string.
- It should extract features that are useful to playing the game.
- Ensure that the perception module is working correctly in that the intended information in the perception code is correctly being presented in the features from perception module section.

Your summary should be grounded in the episode outcome above:

If the episode was a FAILURE (0% progress or death):
- What was the PRIMARY CAUSE of failure? Trace back from the end state to identify the critical mistake(s).
- What beliefs led to bad decisions? 
- What beliefs were missing that would have prevented this outcome?
- Did the perception module include any misleading information that led to this outcome?
- Was there information that the perception module could have included that would have prevented this outcome?

If the episode was a PARTIAL SUCCESS (some progress but not completed):
- What allowed progress to be made? These patterns should be preserved.
- What prevented full completion? Identify the specific bottleneck or mistake.
- Did the perception module help with the successful parts? What did it do incorrectly or what was it missing for the unsuccessful parts?

If the episode was a SUCCESS:
- What key decisions led to success? If not already present, what beliefs can we infer about the world from this?
- What information from perception (if any) was most valuable?
- Was there unnecessary inefficiency that could be improved?

Provide a summary highlighting:
- Root cause analysis: Why did the episode end this way?
- Belief analysis: What beliefs can we infer from the trajectory, especially those that may lead to a positive outcome. Were there beliefs that were incorrect or misleading?
- Perception analysis: What information was presented in the explicit features from perception module section. What part of that information was helpful, what information was misleading / incorrect and what missing information could have helped if extracted by the perception module?
- Perception correctness: Regardless of the outcome of the episode, verify whether the perception module is working correctly. Check that the output of the perception module is correctly mapping the corresponding direct game observation into the intended features.
- Patterns to preserve: What worked well and should NOT be changed?

Format your response in XML style as -
<think>
Analyze the trajectory in light of the episode outcome. Focus on causality - what led to this specific result?
</think>
<summary>
Summary with the sections above clearly addressed
</summary>
"""

    # Build input for LLM
    input_data = build_llm_input(prompt)

    # Call LLM asynchronously
    logging.info(f"Calling LLM to summarize instructions/perception for: {trajectory_path}")
    logging.info(f"Instructions/perception summary prompt:\n{prompt}")
    if config.client.client_name == "vllm":
        model_name = f"hosted_vllm/{config.client.model_id}"
    else:
        model_name = f"{config.client.client_name}/{config.client.model_id}"
    response = await asyncio.to_thread(
        litellm.responses,
        model=model_name,
        input=input_data,
        num_retries=5,
    )

    # Extract response text
    response_text = extract_llm_response_text(response)
    logging.info(f"Instructions/perception summary LLM response for {trajectory_path}:\n{response_text}")

    cost = _get_response_cost(response, config.client.model_id)

    # Extract summary from XML
    response_dict = extract_xml_kv(response_text, ["summary"])

    if not validate_response_fields(response_dict, response_text, ["summary"]):
        return "", cost
    else:
        return response_dict["summary"], cost


async def _get_experiment_summary_async(
    config: DictConfig,
    experiment: str,
    outcome_header: str,
    traj_text: str,
    trajectory_path: str,
) -> str:
    """Get a summary focused on experiment evaluation (async).

    Args:
        config: Configuration containing model information
        experiment: The experiment that was being tested in this episode
        outcome_header: Formatted episode outcome header
        traj_text: Trajectory text content
        trajectory_path: Path to the trajectory file (for logging)

    Returns:
        Summary text focused on experiment evaluation
    """
    prompt = f"""We are playing a game and testing a specific experiment about how it works.

We were testing the following experiment in this episode:
=== EXPERIMENT ===
{experiment}
=== END EXPERIMENT ===

Here is the episode outcome and trajectory:

{outcome_header}
{traj_text}

Analyse whether the above trajectory provides enough evidence to validate or invalidate the given experiment.

Format your response in XML style as -
<think>
Analyse whether we can validate or invalidate the experiment from the given trajectory.
</think>
<summary>
A summary of the reasons behind whether the given experiment is correct or not, or an explanation of insufficient evidence for a conclusion.
</summary>
"""

    # Build input for LLM
    input_data = build_llm_input(prompt)

    # Call LLM asynchronously
    logging.info(f"Calling LLM to summarize experiment for: {trajectory_path}")
    logging.info(f"Experiment summary prompt:\n{prompt}")
    if config.client.client_name == "vllm":
        model_name = f"hosted_vllm/{config.client.model_id}"
    else:
        model_name = f"{config.client.client_name}/{config.client.model_id}"
    response = await asyncio.to_thread(
        litellm.responses,
        model=model_name,
        input=input_data,
        num_retries=5,
    )

    # Extract response text
    response_text = extract_llm_response_text(response)
    logging.info(f"Experiment summary LLM response for {trajectory_path}:\n{response_text}")

    cost = _get_response_cost(response, config.client.model_id)

    # Extract summary from XML
    response_dict = extract_xml_kv(response_text, ["summary"])

    if not validate_response_fields(response_dict, response_text, ["summary"]):
        return "", cost
    else:
        return response_dict["summary"], cost


async def get_episode_summary_async(
    config: DictConfig,
    instruction: str,
    perception: str,
    trajectory_path: str,
    experiment: str = None,
) -> str:
    """Get a summary of an episode trajectory using LLM (async).

    This function makes separate LLM calls for:
    1. Instructions and perception evaluation
    2. Experiment evaluation (if experiment is provided)

    Args:
        config: Configuration containing model information
        instruction: Current beliefs/instructions about the game
        perception: Current perception module code
        trajectory_path: Path to the trajectory file
        experiment: Optional experiment that was being tested in this episode

    Returns:
        Summary text extracted from LLM response(s)
    """
    # Get episode outcome header from JSON metadata
    outcome_header = get_episode_outcome_header(trajectory_path)

    traj_text = Path(trajectory_path).read_text()
    traj_text = trim_to_model_context_lim(traj_text, config.client.model_id)

    # Call for instructions and perception summary
    instructions_perception_summary, ip_cost = await _get_instructions_perception_summary_async(
        config, instruction, perception, outcome_header, traj_text, trajectory_path
    )

    total_cost = ip_cost

    # If experiment is provided, make a separate call for experiment summary
    if experiment:
        experiment_summary, h_cost = await _get_experiment_summary_async(
            config, experiment, outcome_header, traj_text, trajectory_path
        )
        total_cost += h_cost
        # Combine both summaries
        combined_summary = f"""=== INSTRUCTIONS AND PERCEPTION ANALYSIS ===
{instructions_perception_summary}

=== EXPERIMENT ANALYSIS ===
{experiment_summary}"""
        return combined_summary, total_cost
    else:
        return instructions_perception_summary, total_cost


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


def improve_step(
    config: DictConfig,
    base_beliefs: str,
    perception: str,
    output_dir: str,
    previous_experiments: list[str],
    default_knowledge: str,
    num_experiments: int = 0,
    rollout_results: dict[str, dict] | None = None,
) -> tuple[str, str, list[str], float]:
    """Improve step: Evaluate, Update Beliefs/Perception, optionally Generate New Experiments.

    When num_experiments > 0, the LLM is asked to generate new experiments to test.
    When num_experiments == 0, only beliefs and perception are updated (no experiment generation).

    Args:
        config: Configuration containing model information
        base_beliefs: Current beliefs/instructions
        perception: Current perception module
        output_dir: Directory containing rollout results
        previous_experiments: List of experiments tested in the last step
        default_knowledge: Default knowledge string to include in prompt
        num_experiments: Number of new experiments to generate (0 = no generation)
        rollout_results: Results from run_explore_rollouts, including any errors

    Returns:
        Tuple of (updated_beliefs, updated_perception, new_experiments, total_improve_cost).
        new_experiments is [] when num_experiments == 0.
    """
    generate_experiments = num_experiments > 0

    async def get_all_summaries():
        """Get all episode summaries in parallel."""
        episode_tasks = []

        for episode_path in Path(output_dir).rglob("*.csv"):
            try:
                rel_path = episode_path.relative_to(output_dir)
                run_name = rel_path.parts[0]  # e.g., baseline_0, experiment_1

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

    # Build prompt — experiment generation section is conditional
    if generate_experiments:
        experience_preamble = "We have collected new experience."
        task_section = f"""Your task is to:
1. Analyze the results. If experiments were tested, determine if they were confirmed or refuted.
2. Update our beliefs about the game based on confirmed knowledge.
3. Update the perception module to make sure it is correct and that it extracts better features from the direct game observation.
4. Generate {num_experiments} NEW experiments to test in the next step.
   - Experiments should be specific, actionable strategies or mechanics to test.
   - They should help us achieve the main goal."""
        conservatism_note = ""
        think_tag = "<think>\nAnalyze results, evaluate experiments, determine belief updates, design perception improvements, and brainstorm new experiments.\n</think>"
        experiments_xml = f"""<new_experiments>
EXPERIMENT 1: [First experiment to test]

EXPERIMENT 2: [Second experiment to test]
...
(Generate exactly {num_experiments} experiments)
</new_experiments>"""
    else:
        experience_preamble = "We have collected new experience by attempting to play the game with certain experiments in mind."
        task_section = """Your task is to:
1. Analyze the results.
2. Update our beliefs about the game based on confirmed knowledge.
3. Update the perception module to make sure it is correct and that it extracts better features from the direct game observation."""
        conservatism_note = "\nSince we are only evaluating experiments, only update the beliefs or the perception if we have learned anything new from the collected experience. Do not update them if we have not learned anything new."
        think_tag = "<think>\nAnalyze results, evaluate experiments, determine belief updates, and design perception improvements.\n</think>"
        experiments_xml = ""

    base_prompt = f"""We are playing a game and trying to figure out how it works.
Current beliefs about the game:
{base_beliefs if base_beliefs else "(empty - no beliefs yet)"}

The agent also receives the following default instructions/knowledge by default:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

Current perception module:
{perception if perception else "(empty - no perception module yet)"}

{experience_preamble}
{evidence_section}

{task_section}

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
{conservatism_note}

Format your response in XML style as:
{think_tag}
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
{experiments_xml}
"""

    # Setup model name
    if config.client.client_name == "vllm":
        model_name = f"hosted_vllm/{config.client.model_id}"
    else:
        model_name = f"{config.client.client_name}/{config.client.model_id}"

    # Determine which XML fields to extract
    xml_fields = ["updated_beliefs", "perception"]
    if generate_experiments:
        xml_fields.append("new_experiments")

    # Retry loop for perception validation
    max_retries = 3
    perception_error = None
    updated_beliefs = base_beliefs
    updated_perception = perception
    new_experiments = []
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
        response_dict = extract_xml_kv(response_text, xml_fields)
        validate_response_fields(response_dict, response_text, xml_fields)

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

        # Process experiments (do this on every attempt so we have them even if perception fails)
        if generate_experiments and "new_experiments" in response_dict:
            experiments_text = response_dict["new_experiments"].strip()
            experiment_pattern = r'EXPERIMENT\s*\d+\s*:\s*(.+?)(?=EXPERIMENT\s*\d+\s*:|$)'
            matches = re.findall(experiment_pattern, experiments_text, re.DOTALL | re.IGNORECASE)
            new_experiments = [m.strip() for m in matches if m.strip()]

            # Fallback split
            if not new_experiments:
                new_experiments = [b.strip() for b in experiments_text.split('\n\n') if b.strip()]

    # Process experiments from final response if not already processed
    if generate_experiments and not new_experiments and "new_experiments" in response_dict:
        experiments_text = response_dict["new_experiments"].strip()
        experiment_pattern = r'EXPERIMENT\s*\d+\s*:\s*(.+?)(?=EXPERIMENT\s*\d+\s*:|$)'
        matches = re.findall(experiment_pattern, experiments_text, re.DOTALL | re.IGNORECASE)
        new_experiments = [m.strip() for m in matches if m.strip()]

        # Fallback split
        if not new_experiments:
            new_experiments = [b.strip() for b in experiments_text.split('\n\n') if b.strip()]

    # Limit to requested num
    if generate_experiments:
        new_experiments = new_experiments[:num_experiments]

    logging.info(f"Updated beliefs:\n{updated_beliefs}")
    logging.info(f"Updated perception:\n{updated_perception}")
    logging.info(f"Generated {len(new_experiments)} new experiments")

    total_improve_cost = summary_cost + improve_call_cost
    return updated_beliefs, updated_perception, new_experiments, total_improve_cost

