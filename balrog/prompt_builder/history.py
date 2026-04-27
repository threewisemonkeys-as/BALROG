from collections import deque
from typing import List, Optional


class Message:
    """Represents a conversation message with role, content, and optional attachment."""

    def __init__(self, role: str, content: str, attachment: Optional[object] = None):
        self.role = role  # 'system', 'user', 'assistant'
        self.content = content  # String content of the message
        self.attachment = attachment

    def __repr__(self):
        return f"Message(role={self.role}, content={self.content}, attachment={self.attachment})"


class HistoryPromptBuilder:
    """Builds a prompt with a history of observations, actions, and reasoning.

    Maintains a configurable history of text, images, and chain-of-thought reasoning to
    construct prompt messages for conversational agents.
    """

    def __init__(
        self,
        max_text_history: int = 16,
        max_image_history: int = 1,
        system_prompt: Optional[str] = None,
        max_cot_history: int = 1,
    ):
        self.max_text_history = max_text_history
        self.max_image_history = max_image_history
        self.max_history = max(max_text_history, max_image_history)
        self.system_prompt = system_prompt
        self._events = deque(maxlen=self.max_history * 2)  # Stores observations and actions
        self._last_short_term_obs = None  # To store the latest short-term observation
        self.previous_reasoning = None
        self.max_cot_history = max_cot_history

    def update_instruction_prompt(self, instruction: str):
        """Set the system-level instruction prompt."""
        self.system_prompt = instruction

    def update_observation(self, obs: dict):
        """Add an observation to the prompt history, which can include text, an image, or both."""
        long_term_context = obs["text"].get("long_term_context", "")
        self._last_short_term_obs = obs["text"].get("short_term_context", "")
        text = long_term_context

        raw_images = obs.get("images", None)
        if raw_images:
            images = [
                item
                for item in raw_images
                if isinstance(item, dict) and item.get("image") is not None
            ]
        else:
            image = obs.get("image", None)
            images = [{"label": "Image observation", "image": image}] if image is not None else []

        # Add observation to events
        self._events.append(
            {
                "type": "observation",
                "text": text,
                "images": images,
            }
        )

    def update_action(self, action: str):
        """Add an action to the prompt history, including reasoning if available."""
        self._events.append(
            {
                "type": "action",
                "action": action,
                "reasoning": self.previous_reasoning,
            }
        )

    def update_reasoning(self, reasoning: str):
        """Set the reasoning text to be included with subsequent actions."""
        self.previous_reasoning = reasoning

    def reset(self):
        """Clear the event history."""
        self._events.clear()

    def get_prompt(self, icl_episodes=False) -> List[Message]:
        """Generate a list of Message objects representing the prompt.

        Returns:
            List[Message]: Messages constructed from the event history.
        """
        messages = []

        if self.system_prompt and not icl_episodes:
            messages.append(Message(role="user", content=self.system_prompt))

        # Determine which text observations to include
        text_needed = self.max_text_history
        for event in reversed(self._events):
            if event["type"] == "observation":
                if text_needed > 0 and event.get("text") is not None:
                    event["include_text"] = True
                    text_needed -= 1
                else:
                    event["include_text"] = False

        # Determine which image observations to include. Counts observation
        # events with images, mirroring max_text_history semantics; events with
        # multiple images (e.g. AutumnBench planning's current+goal) consume
        # one slot, not one per image.
        events_needed = self.max_image_history
        for event in reversed(self._events):
            if event["type"] == "observation":
                images = event.get("images") or []
                if events_needed > 0 and images:
                    event["include_image"] = True
                    event["included_images"] = images
                    events_needed -= 1
                else:
                    event["include_image"] = False
                    event["included_images"] = []

        # determine the reasoning to include
        reasoning_needed = self.max_cot_history
        for event in reversed(self._events):
            if event["type"] == "action":
                if reasoning_needed > 0 and event.get("reasoning") is not None:
                    reasoning_needed -= 1
                else:
                    event["reasoning"] = None

        # Process events to create messages
        for idx, event in enumerate(self._events):
            if event["type"] == "observation":
                message_parts = []

                if idx == len(self._events) - 1:
                    message_parts.append("Current Observation:")
                    if self._last_short_term_obs:
                        message_parts.append(self._last_short_term_obs)
                else:
                    message_parts.append("Observation:")

                if event.get("include_text", False):
                    message_parts.append(event["text"])
                    
                attachments = []
                if event.get("include_image", False):
                    included_images = event.get("included_images", [])
                    attachments = [item["image"] for item in included_images]
                    labels = [item.get("label", "Image observation") for item in included_images]
                    if len(labels) == 1:
                        message_parts.append(f"{labels[0]} image provided.")
                    elif labels:
                        message_parts.append(
                            "Images provided: " + ", ".join(f"{label}" for label in labels) + "."
                        )

                content = "\n".join(message_parts)
                attachment = attachments if len(attachments) > 1 else (attachments[0] if attachments else None)
                message = Message(role="user", content=content, attachment=attachment)

                # Clean up temporary flags
                for flag in ["include_text", "include_image", "included_images"]:
                    if flag in event:
                        del event[flag]
            elif event["type"] == "action":
                if event.get("reasoning") is not None:
                    content = "Previous plan:\n" + event["reasoning"]
                else:
                    content = event["action"]
                message = Message(role="assistant", content=content)
            messages.append(message)

        return messages
