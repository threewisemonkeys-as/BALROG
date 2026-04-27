import copy
import re

from balrog.agents.base import BaseAgent
from balrog.client import LLMClientWrapper


class PlanMemoryAgent(BaseAgent):
    """An agent that performs actions using chain-of-thought reasoning and maintains a plan across steps."""

    def __init__(self, client_factory: LLMClientWrapper, prompt_builder, config):
        """Initialize the PlanMemoryAgent with a client, prompt builder, and configuration.

        Args:
            client_factory (LLMClientWrapper): A factory for creating the LLM client instance.
            prompt_builder (PromptBuilder): Object to build prompts for the agent.
            config: Configuration object containing settings for the agent.
        """
        super().__init__(client_factory, prompt_builder)
        self.remember_cot = config.agent.remember_cot
        self.instruction_text = config.eval.get("instruction_prompt", None)
        self.previous_plan = None  # Store the previous plan

    def act(self, obs, prev_action=None):
        """Generate the next action using chain-of-thought reasoning based on the current observation.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the final selected action.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        if self.instruction_text is not None:
            messages[-1].content += "\n\n" + f"""
Tips -
{self.instruction_text}
            """.strip()

        # Build the instruction with previous plan if available
        instruction = ""

        if self.previous_plan is not None:
            instruction += f"""
Previous plan:
<previous_plan>
{self.previous_plan}
</previous_plan>

"""

        instruction += """
First create (if not present) or update your plan from the previous steps and present the updated plan in -
<plan>
updated plan
</plan>

Then think about what the best course of action might be. Present this in the following format -
<think> thinking about what to do next </think>

Finally you must choose exactly one of the listed actions and output it strictly in the following format -
<|ACTION|>YOUR_CHOSEN_ACTION<|END|>

Replace YOUR_CHOSEN_ACTION with the chosen action.
Keep the plan and thinking very brief.
        """.strip()

        messages[-1].content += "\n\n" + instruction

        self.last_messages = messages
        # Generate the CoT reasoning
        cot_reasoning = self.client.generate(messages)

        # Extract the plan from the current response
        self._extract_and_store_plan(cot_reasoning)

        # Extract the final answer from the CoT reasoning
        final_answer = self._extract_final_answer(cot_reasoning)
        self.prompt_builder.update_reasoning(final_answer.reasoning)

        return final_answer

    def _extract_and_store_plan(self, reasoning):
        """Extract the plan section from the reasoning and store it for the next step.

        Args:
            reasoning (LLMResponse): The response containing the plan section.
        """
        completion_text = reasoning.completion
        match = re.search(r"<plan>(.*?)</plan>", completion_text, re.DOTALL | re.IGNORECASE)
        if match:
            self.previous_plan = match.group(1).strip()
        # If no plan is found, we keep the previous plan (or None if this is the first step)

    def _extract_final_answer(self, reasoning):
        """Extract the final action from the chain-of-thought reasoning response.

        Args:
            reasoning (LLMResponse): The response containing CoT reasoning and action.

        Returns:
            LLMResponse: The response with the extracted final action in `completion`
                         and the entire chain-of-thought in `reasoning`.
        """
        # Make a copy so we don't mutate the original
        final_answer = copy.deepcopy(reasoning)

        # Store the entire chain-of-thought (raw completion) in `reasoning`
        final_answer = final_answer._replace(reasoning=reasoning.completion)

        # Now parse the strict action format: <|ACTION|> ... <|END|>
        completion_text = reasoning.completion
        match = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", completion_text, re.DOTALL)
        if match:
            extracted_action = match.group(1).strip()
        else:
            # Fallback to the entire completion if not matched
            extracted_action = "Failed to obtain a valid action from the reasoning."

        # Replace the final `completion` with only the extracted action
        final_answer = final_answer._replace(completion=extracted_action)

        return final_answer
