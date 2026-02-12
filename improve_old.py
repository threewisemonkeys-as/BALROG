import asyncio
import json
import logging
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


def improve_both(
    config: DictConfig,
    instruction: str,
    perception: str,
    output_dir: str,
) -> tuple[str, str, str]:
    """Improve both beliefs and perception module based on episode trajectories using LLM.

    Args:
        config: Configuration containing model information
        instruction: Current beliefs/instructions about the game
        perception: Current perception module code
        output_dir: Directory containing episode trajectory files

    Returns:
        Tuple of (updated_beliefs, updated_perception, synthesis_prompt)
    """

    async def get_all_summaries():
        """Get all episode summaries in parallel."""
        episode_paths = list(Path(output_dir).rglob("*.csv"))
        logging.info(f"Processing {len(episode_paths)} episodes in parallel")

        tasks = [
            get_episode_summary_async(config, instruction, perception, str(episode_path))
            for episode_path in episode_paths
        ]
        return await asyncio.gather(*tasks)

    # Get summaries for all episodes in parallel
    ep_results = asyncio.run(get_all_summaries())

    # Combine all summaries
    ep_summaries_str = ""
    for ep_sum, _cost in ep_results:
        ep_summaries_str += f"Episode Summary:\n{ep_sum.strip()}\n\n"

    # Build prompt for updating both beliefs and perception
    prompt = f"""We are playing a game and trying to figure out how it works.
Currently we have the following list of beliefs about the game -

{instruction}

We also have a perception module that processes the game observation text and outputs text. The perception module currently looks like this -

{perception if perception else "(empty - no perception module yet)"}

We have played the game using these beliefs and perception with the following results -

{ep_summaries_str}

Think about how we need to update both our beliefs AND our perception module given this information.

For the beliefs: Only make updates based on the results provided above, do not assume anything about the game.

For the perception module:
- It should be a Python function that takes the observation text as input (which includes the ASCII map, messages, inventory, etc.) and returns text describing useful features
- The function signature should be:
```python
def perceive(observation_text: str) -> str:
    # Process observation and return textual description of useful features
    pass
```
- The input `observation_text` is a multi-line string containing labeled sections like "message:", "cursor:", "map:", etc.
- You can parse this text to extract the ASCII map and other information
- The perception module should extract information that is relevant to playing the game well
- It might identify patterns, count objects, detect dangers, compute spatial relationships, or highlight important features
- Keep it concise and focused on actionable information
- The perception output will be presented before the observation in its own section

Format your response in XML style as -
<think>
Think about:
1. What went wrong or right in the episodes
2. How our beliefs should be updated
3. What information from the observation would be most useful to extract
4. How to improve or create the perception module
</think>
<beliefs>
- updated belief
- updated belief
...
</beliefs>
<perception>
```python
def perceive(observation_text: str) -> str:
    # Your implementation here
    pass
```
</perception>"""

    logging.info(f"Final synthesis prompt for improve_both:\n{prompt}")

    # Build input for LLM
    input_data = build_llm_input(prompt)

    # Call LLM
    logging.info(f"Calling LLM to generate updated beliefs and perception")
    if config.client.client_name == "vllm":
        model_name = f"hosted_vllm/{config.client.model_id}"
    else:
        model_name = f"{config.client.client_name}/{config.client.model_id}"
    response = litellm.responses(
        model=model_name,
        input=input_data,
        num_retries=5,
    )

    # Extract response text
    response_text = extract_llm_response_text(response)

    # Extract beliefs and perception from XML
    response_dict = extract_xml_kv(response_text, ["beliefs", "perception"])
    validate_response_fields(response_dict, response_text, ["beliefs", "perception"])

    if "beliefs" in response_dict:
        updated_beliefs = response_dict["beliefs"].strip()
    else:
        updated_beliefs = instruction

    if "perception" in response_dict:
        updated_perception = response_dict["perception"].strip()
    else:
        updated_perception = perception

    # Strip markdown code fence markers from perception
    if updated_perception.startswith("```python"):
        updated_perception = updated_perception[len("```python"):].strip()
    elif updated_perception.startswith("```"):
        updated_perception = updated_perception[len("```"):].strip()

    if updated_perception.endswith("```"):
        updated_perception = updated_perception[:-len("```")].strip()

    logging.info(f"Updated beliefs:\n{updated_beliefs}")
    logging.info(f"Updated perception module:\n{updated_perception}")

    return updated_beliefs, updated_perception, prompt


def improve_perception_only(
    config: DictConfig,
    instruction: str,
    perception: str,
    output_dir: str,
) -> tuple[str, str]:
    """Improve only the perception module based on episode trajectories, keeping beliefs fixed.

    Args:
        config: Configuration containing model information
        instruction: Fixed beliefs/instructions about the game (not updated)
        perception: Current perception module code
        output_dir: Directory containing episode trajectory files

    Returns:
        Tuple of (updated_perception, synthesis_prompt)
    """

    async def get_all_summaries():
        """Get all episode summaries in parallel."""
        episode_paths = list(Path(output_dir).rglob("*.csv"))
        logging.info(f"Processing {len(episode_paths)} episodes in parallel")

        tasks = [
            get_episode_summary_async(config, instruction, perception, str(episode_path))
            for episode_path in episode_paths
        ]
        return await asyncio.gather(*tasks)

    # Get summaries for all episodes in parallel
    ep_results = asyncio.run(get_all_summaries())

    # Combine all summaries
    ep_summaries_str = ""
    for ep_sum, _cost in ep_results:
        ep_summaries_str += f"Episode Summary:\n{ep_sum.strip()}\n\n"

    # Build prompt for updating only perception
    prompt = f"""We are playing a game and trying to figure out how it works.
We have the following beliefs about the game -

{instruction}

We also have a perception module that processes the game observation text and outputs text. The perception module currently looks like this -

{perception if perception else "(empty - no perception module yet)"}

We have played the game using these beliefs and perception with the following results -

{ep_summaries_str}

Think about how we need to update our perception module given this information. Do not consider how the beliefs might be updated at all.

For the perception module:
- It should be a Python function that takes the observation text as input (which includes the ASCII map, messages, inventory, etc.) and returns text describing useful features
- The function signature should be:
```python
def perceive(observation_text: str) -> str:
    # Process observation and return textual description of useful features
    pass
```
- The input `observation_text` is a multi-line string containing labeled sections like "message:", "cursor:", "map:", etc.
- You can parse this text to extract the ASCII map and other information
- The perception module should extract information that is relevant to playing the game well according to our fixed beliefs
- It might identify patterns, count objects, detect dangers, compute spatial relationships, or highlight important features
- Keep it concise and focused on actionable information
- The perception output will be presented before the observation as its own section

Format your response in XML style as -
<think>
Think about:
1. What went wrong or right in the episodes
2. What information from the observation would be most useful to extract given our fixed beliefs
3. How to improve or create the perception module
</think>
<perception>
```python
def perceive(observation_text: str) -> str:
    # Your implementation here
    pass
```
</perception>"""

    logging.info(f"Final synthesis prompt for improve_perception_only:\n{prompt}")

    # Build input for LLM
    input_data = build_llm_input(prompt)

    # Call LLM
    logging.info(f"Calling LLM to generate updated perception (beliefs fixed)")
    if config.client.client_name == "vllm":
        model_name = f"hosted_vllm/{config.client.model_id}"
    else:
        model_name = f"{config.client.client_name}/{config.client.model_id}"
    response = litellm.responses(
        model=model_name,
        input=input_data,
        num_retries=5,
    )

    # Extract response text
    response_text = extract_llm_response_text(response)

    # Extract perception from XML
    response_dict = extract_xml_kv(response_text, ["perception"])
    validate_response_fields(response_dict, response_text, ["perception"])

    if "perception" in response_dict:
        updated_perception = response_dict["perception"].strip()
    else:
        updated_perception = perception

    # Strip markdown code fence markers from perception
    if updated_perception.startswith("```python"):
        updated_perception = updated_perception[len("```python"):].strip()
    elif updated_perception.startswith("```"):
        updated_perception = updated_perception[len("```"):].strip()

    if updated_perception.endswith("```"):
        updated_perception = updated_perception[:-len("```")].strip()

    logging.info(f"Updated perception module:\n{updated_perception}")

    return updated_perception, prompt

