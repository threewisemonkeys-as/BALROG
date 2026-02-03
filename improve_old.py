import asyncio
import logging
from pathlib import Path

import litellm
from omegaconf import DictConfig

from llm_utils import build_llm_input, extract_llm_response_text, extract_xml_kv, validate_response_fields


def trim_to_model_context_lim(text: str, model_name: str, buffer: int = 0, prefix: bool = True) -> str:
    if "qwen" in model_name.lower():
        return text[-100000:]
    else:
        return text[-350000:]


async def get_episode_summary_async(
    config: DictConfig,
    instruction: str,
    perception: str,
    trajectory_path: str
) -> str:
    """Get a summary of an episode trajectory using LLM (async).

    Args:
        config: Configuration containing model information
        instruction: Current beliefs/instructions about the game
        perception: Current perception module code
        trajectory_path: Path to the trajectory file

    Returns:
        Summary text extracted from LLM response
    """
    traj_text = Path(trajectory_path).read_text()
    traj_text = trim_to_model_context_lim(traj_text, config.client.model_id)

    prompt = f"""We are playing a game and trying to figure out how it works.
Currently we have the following list of beliefs about the game -

{instruction}

We are also using the following perception module to process game observations -

{perception if perception else "(empty - no perception module)"}

We have played the game using these beliefs and perception resulting in the following trajectory -

{traj_text}

Provide a short summary of the trajectory, highlighting:
- Key decisions made and their outcomes
- Mistakes made if any
- Whether the perception module (if present) provided useful information or if it's missing important features
- What information would have been helpful to have from the observations

Format your response in XML style as -
<think>
Think about the trajectory followed when playing the game and the quality of perceptual information.
</think>
<summary>
Summary as requested
</summary>
"""

    # Build input for LLM
    input_data = build_llm_input(prompt)

    # Call LLM asynchronously
    logging.info(f"Calling LLM to summarize trajectory: {trajectory_path}")
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

    # Extract summary from XML
    response_dict = extract_xml_kv(response_text, ["summary"])


    if not validate_response_fields(response_dict, response_text, ["summary"]):
        return ""
    else:
        return response_dict["summary"]


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
    ep_summaries = asyncio.run(get_all_summaries())

    # Combine all summaries
    ep_summaries_str = ""
    for ep_sum in ep_summaries:
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
- The perception output will be prepended to the observation as "perceptual features:" section

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
    ep_summaries = asyncio.run(get_all_summaries())

    # Combine all summaries
    ep_summaries_str = ""
    for ep_sum in ep_summaries:
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
- The perception output will be prepended to the observation as "perceptual features:" section

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


async def _get_perception_improvement_req_async(
    config: DictConfig,
    instruction: str,
    perception: str,
    trajectory_path: str
) -> str:
    """Get improvement requirements for the perception module from a single trajectory.

    Args:
        config: Configuration containing model information
        instruction: Current beliefs/instructions about the game
        perception: Current perception module code
        trajectory_path: Path to the trajectory file

    Returns:
        Paragraph describing suggested improvements for the perception module
    """
    traj_text = Path(trajectory_path).read_text()
    traj_text = trim_to_model_context_lim(traj_text, config.client.model_id)

    prompt = f"""We are playing a game and trying to improve our perception module that processes game observations.

Current beliefs about the game:
{instruction if instruction else "(empty - no beliefs yet)"}

Current perception module:
{perception if perception else "(empty - no perception module yet)"}

The agent played using the above beliefs and perception, resulting in the following trajectory:

{traj_text}

Analyze this trajectory and identify ways the perception module could be improved to help the agent perform better. Focus on:
- What information from the observation was the agent missing or misinterpreting?
- What patterns or features should the perception module extract that it currently doesn't?
- Did the perception module provide irrelevant or misleading information?
- What spatial relationships, object counts, or danger indicators would have been helpful?

Format your response in XML style as:
<think>
Analyze the trajectory and what went wrong or right related to perception.
</think>
<perception_improvements>
A short paragraph (2-4 sentences) describing specific improvements needed for the perception module based on this trajectory.
</perception_improvements>
"""

    input_data = build_llm_input(prompt)

    logging.info(f"Getting perception improvement requirements from: {trajectory_path}")
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

    response_text = extract_llm_response_text(response)
    response_dict = extract_xml_kv(response_text, ["perception_improvements"])

    if not validate_response_fields(response_dict, response_text, ["perception_improvements"]):
        return ""
    return response_dict["perception_improvements"].strip()


async def _get_beliefs_improvement_req_async(
    config: DictConfig,
    instruction: str,
    perception: str,
    trajectory_path: str
) -> str:
    """Get improvement requirements for the beliefs from a single trajectory.

    Args:
        config: Configuration containing model information
        instruction: Current beliefs/instructions about the game
        perception: Current perception module code
        trajectory_path: Path to the trajectory file

    Returns:
        Paragraph describing suggested improvements for the beliefs
    """
    traj_text = Path(trajectory_path).read_text()
    traj_text = trim_to_model_context_lim(traj_text, config.client.model_id)

    prompt = f"""We are playing a game and trying to improve our beliefs/instructions that guide the agent's decisions.

Current beliefs about the game:
{instruction if instruction else "(empty - no beliefs yet)"}

Current perception module:
{perception if perception else "(empty - no perception module yet)"}

The agent played using the above beliefs and perception, resulting in the following trajectory:

{traj_text}

Analyze this trajectory and identify ways the beliefs could be improved to help the agent perform better. Focus on:
- What incorrect assumptions did the agent make about game mechanics?
- What strategies or heuristics would have led to better decisions?
- What rules about the game did the agent seem to violate or not know?
- What priorities should the agent have that it currently lacks?

Important: Only suggest improvements based on evidence from this trajectory. Do not assume things about the game that aren't demonstrated.

Format your response in XML style as:
<think>
Analyze the trajectory and what went wrong or right related to the agent's beliefs and decision-making.
</think>
<beliefs_improvements>
A short paragraph (2-4 sentences) describing specific improvements needed for the beliefs based on this trajectory.
</beliefs_improvements>
"""

    input_data = build_llm_input(prompt)

    logging.info(f"Getting beliefs improvement requirements from: {trajectory_path}")
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

    response_text = extract_llm_response_text(response)
    response_dict = extract_xml_kv(response_text, ["beliefs_improvements"])

    if not validate_response_fields(response_dict, response_text, ["beliefs_improvements"]):
        return ""
    return response_dict["beliefs_improvements"].strip()


def improve_both_deep(
    config: DictConfig,
    instruction: str,
    perception: str,
    output_dir: str,
) -> tuple[str, str, str]:
    """Improve both beliefs and perception using deep trajectory analysis.

    This function performs a more thorough improvement process:
    1. For each trajectory, gather specific improvement requirements for the perception module
    2. For each trajectory, gather specific improvement requirements for the beliefs
    3. Synthesize all requirements to generate improved beliefs and perception

    Args:
        config: Configuration containing model information
        instruction: Current beliefs/instructions about the game
        perception: Current perception module code
        output_dir: Directory containing episode trajectory files

    Returns:
        Tuple of (updated_beliefs, updated_perception, synthesis_prompt)
    """
    episode_paths = list(Path(output_dir).rglob("*.csv"))
    logging.info(f"improve_both_deep: Processing {len(episode_paths)} episodes")

    async def gather_all_improvement_reqs():
        """Gather all improvement requirements in parallel."""
        # Create tasks for perception improvements
        perception_tasks = [
            _get_perception_improvement_req_async(config, instruction, perception, str(ep))
            for ep in episode_paths
        ]
        # Create tasks for beliefs improvements
        beliefs_tasks = [
            _get_beliefs_improvement_req_async(config, instruction, perception, str(ep))
            for ep in episode_paths
        ]

        # Run all tasks in parallel
        all_results = await asyncio.gather(*perception_tasks, *beliefs_tasks)

        # Split results
        n = len(episode_paths)
        perception_reqs = all_results[:n]
        beliefs_reqs = all_results[n:]

        return perception_reqs, beliefs_reqs

    # Gather improvement requirements from all trajectories
    perception_reqs, beliefs_reqs = asyncio.run(gather_all_improvement_reqs())

    # Format perception improvement requirements
    perception_reqs_str = ""
    for i, req in enumerate(perception_reqs, 1):
        if req:
            perception_reqs_str += f"Trajectory {i}:\n{req}\n\n"

    # Format beliefs improvement requirements
    beliefs_reqs_str = ""
    for i, req in enumerate(beliefs_reqs, 1):
        if req:
            beliefs_reqs_str += f"Trajectory {i}:\n{req}\n\n"

    logging.info(f"Collected perception improvement requirements:\n{perception_reqs_str}")
    logging.info(f"Collected beliefs improvement requirements:\n{beliefs_reqs_str}")

    # Build the synthesis prompt
    prompt = f"""We are playing a game and trying to improve both our beliefs and perception module.

CURRENT BELIEFS:
{instruction if instruction else "(empty - no beliefs yet)"}

CURRENT PERCEPTION MODULE:
{perception if perception else "(empty - no perception module yet)"}

After analyzing multiple game trajectories, we have gathered the following improvement requirements:

=== PERCEPTION MODULE IMPROVEMENT REQUIREMENTS ===
{perception_reqs_str if perception_reqs_str else "(no specific requirements gathered)"}

=== BELIEFS IMPROVEMENT REQUIREMENTS ===
{beliefs_reqs_str if beliefs_reqs_str else "(no specific requirements gathered)"}

Based on these requirements, create improved versions of both the beliefs and perception module.

For the beliefs:
- Address the issues identified in the improvement requirements
- Keep beliefs concise and actionable
- Only include beliefs supported by evidence from the trajectories

For the perception module:
- It should be a Python function with signature: def perceive(observation_text: str) -> str
- The input `observation_text` is a multi-line string with labeled sections like "message:", "cursor:", "map:", etc.
- Address the issues identified in the improvement requirements
- Extract information that is relevant to playing the game well
- Keep it concise and focused on actionable information

Format your response in XML style as:
<think>
Synthesize the improvement requirements and plan the updates to beliefs and perception.
</think>
<beliefs>
- belief 1
- belief 2
...
</beliefs>
<perception>
```python
def perceive(observation_text: str) -> str:
    # Implementation here
    pass
```
</perception>
"""

    logging.info(f"Final synthesis prompt for improve_both_deep:\n{prompt}")

    input_data = build_llm_input(prompt)

    logging.info("Calling LLM to synthesize improved beliefs and perception")
    if config.client.client_name == "vllm":
        model_name = f"hosted_vllm/{config.client.model_id}"
    else:
        model_name = f"{config.client.client_name}/{config.client.model_id}"

    response = litellm.responses(
        model=model_name,
        input=input_data,
        num_retries=5,
    )

    response_text = extract_llm_response_text(response)
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
