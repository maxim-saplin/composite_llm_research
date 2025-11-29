"""
Chain of Thought Strategy

This strategy implements a two-step prompting approach:
1. Force the model to generate explicit reasoning in <thinking> tags
2. Feed the thoughts back and generate a final answer

This is NOT the same as Anthropic's "think" tool - see think_tool.py for that.
This is closer to "extended thinking" or chain-of-thought prompting.
"""

from typing import List, Dict, Any
import time

from .base import BaseStrategy
from .think_tool import strip_thinking_tags


class ChainOfThoughtStrategy(BaseStrategy):
    """
    A strategy that forces explicit chain-of-thought reasoning before answering.
    
    Uses two LLM calls:
    1. Generate thoughts with explicit reasoning
    2. Generate final answer based on those thoughts
    """
    
    def execute(
        self,
        messages: List[Dict[str, str]],
        model_config: str,
        optional_params: Dict[str, Any],
        litellm_params: Dict[str, Any],
    ) -> Any:
        target_model = model_config or "gpt-4o"

        trace_recorder = optional_params.get("trace_recorder")
        trace_root_id = optional_params.get("trace_root_node_id")
        strategy_node_id = None
        if trace_recorder and trace_root_id is not None:
            strategy_node_id = trace_recorder.add_node(
                step_type="strategy",
                parent_id=trace_root_id,
                model=target_model,
                content_preview="Chain-of-Thought strategy execution",
            )

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

        start = time.time()
        thought_response = self.simple_completion(
            model=target_model,
            messages=thought_messages,
            stop=["</thinking>"],  # Stop after thinking
            **litellm_params,
        )
        thought_duration = time.time() - start

        thoughts = thought_response.choices[0].message.content
        if "<thinking>" not in thoughts:
            thoughts = f"<thinking>\n{thoughts}\n</thinking>"
        else:
            thoughts = thoughts + "</thinking>"  # Append the stop token we cut off

        if trace_recorder:
            trace_recorder.add_node(
                step_type="llm_call",
                parent_id=strategy_node_id,
                model=target_model,
                role="assistant",
                content_preview=thoughts[:200],
                duration_seconds=thought_duration,
                extra={"stage": "thoughts"},
            )

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

        start = time.time()
        final_response = self.simple_completion(
            model=target_model, messages=final_messages, **litellm_params
        )
        final_duration = time.time() - start

        # Strip any <thinking> tags the model may have echoed back
        if final_response.choices[0].message.content:
            final_response.choices[0].message.content = strip_thinking_tags(
                final_response.choices[0].message.content
            )

        if trace_recorder:
            content = final_response.choices[0].message.content or ""
            trace_recorder.add_node(
                step_type="llm_call",
                parent_id=strategy_node_id,
                model=target_model,
                role="assistant",
                content_preview=content[:200],
                duration_seconds=final_duration,
                extra={"stage": "final"},
            )

        # Store thoughts in reasoning_content field (like o1/DeepSeek models do)
        # Extract just the thinking content without the tags
        raw_thoughts = strip_thinking_tags(
            thoughts.replace("<thinking>", "").replace("</thinking>", "")
        )
        final_response.choices[0].message.reasoning_content = raw_thoughts

        return final_response

