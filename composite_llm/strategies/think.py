from typing import List, Dict, Any
from .base import BaseStrategy
import litellm


class ThinkStrategy(BaseStrategy):
    def execute(
        self,
        messages: List[Dict[str, str]],
        model_config: str,
        optional_params: Dict[str, Any],
        litellm_params: Dict[str, Any],
    ) -> Any:
        target_model = model_config or "gpt-4o"

        # Step 1: Generate Thoughts
        # specific instructions to force thinking
        think_instructions = (
            "You are in a 'Deep Thinking' mode. "
            "Before answering the user's request, you must output a detailed internal monologue "
            "analyzing the problem, checking edge cases, and planning the response. "
            "Enclose your thoughts in <thinking>...</thinking> tags. "
            "Do NOT provide the final answer yet, just the thinking process."
        )

        thought_messages = [m.copy() for m in messages]
        # Inject system instruction
        if thought_messages[0]["role"] == "system":
            thought_messages[0]["content"] += f"\n\n{think_instructions}"
        else:
            thought_messages.insert(
                0, {"role": "system", "content": think_instructions}
            )

        thought_response = litellm.completion(
            model=target_model,
            messages=thought_messages,
            stop=["</thinking>"],  # Stop after thinking
        )

        thoughts = thought_response.choices[0].message.content
        if "<thinking>" not in thoughts:
            thoughts = f"<thinking>\n{thoughts}\n</thinking>"
        else:
            thoughts = thoughts + "</thinking>"  # Append the stop token we cut off

        # Step 2: Generate Final Answer
        # We feed the thoughts back to the model
        final_messages = [m.copy() for m in messages]

        # Add the thoughts as an assistant message (pre-fill) or part of the context
        # Better to add it as a prior turn
        final_messages.append({"role": "assistant", "content": thoughts})
        final_messages.append(
            {
                "role": "user",
                "content": "Now, based on your thoughts, provide the final answer to my original request.",
            }
        )

        final_response = litellm.completion(
            model=target_model, messages=final_messages, **optional_params
        )

        # We might want to prepend the thoughts to the final content if the user wants to see them
        # For now, let's just return the final answer, effectively "hiding" the thinking step
        # (or we could return it in a specific format)

        # Optional: Include thoughts in the response metadata or prepend
        if optional_params.get("include_thoughts", False):
            final_response.choices[
                0
            ].message.content = (
                f"{thoughts}\n\n{final_response.choices[0].message.content}"
            )

        return final_response
