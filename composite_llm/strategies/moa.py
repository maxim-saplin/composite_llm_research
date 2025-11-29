from typing import List, Dict, Any
import time

import litellm

from .base import BaseStrategy


class MoAStrategy(BaseStrategy):
    def execute(
        self,
        messages: List[Dict[str, str]],
        model_config: str,
        optional_params: Dict[str, Any],
        litellm_params: Dict[str, Any],
    ) -> Any:
        """
        Implements Mixture-of-Agents (MoA):
        1. Send query to N proposer models.
        2. Collect responses.
        3. Send all responses + original query to Aggregator model (model_config) to synthesize final answer.
        """

        proposers = optional_params.get(
            "proposers", ["cerebras/llama3.1-8b", "cerebras/qwen-3-32b"]
        )

        trace_recorder = optional_params.get("trace_recorder")
        trace_root_id = optional_params.get("trace_root_node_id")

        strategy_node_id = None
        if trace_recorder and trace_root_id is not None:
            strategy_node_id = trace_recorder.add_node(
                step_type="strategy",
                parent_id=trace_root_id,
                model=model_config,
                content_preview="MoA strategy execution",
            )

        # Step 1: Parallel calls to proposers
        # For simplicity in this synchronous demo, we loop. In production, use asyncio.gather
        proposer_responses = []

        print(f"  [MoA] Querying proposers: {proposers}")

        for p_model in proposers:
            try:
                # We reuse the same messages for proposers
                # Note: We should handle API keys for different providers in real app
                start = time.time()
                resp = self.simple_completion(
                    model=p_model, messages=messages, **litellm_params
                )
                duration = time.time() - start
                content = resp.choices[0].message.content
                proposer_responses.append(f"Model {p_model} suggests:\n{content}")

                if trace_recorder:
                    trace_recorder.add_node(
                        step_type="llm_call",
                        parent_id=strategy_node_id,
                        model=p_model,
                        role="assistant",
                        content_preview=(content or "")[:200],
                        duration_seconds=duration,
                    )
            except Exception as e:
                error_msg = f"Model {p_model} failed: {str(e)}"
                proposer_responses.append(error_msg)
                if trace_recorder:
                    trace_recorder.add_node(
                        step_type="llm_call",
                        parent_id=strategy_node_id,
                        model=p_model,
                        role="assistant",
                        content_preview=error_msg[:200],
                    )

        # Step 2: Aggregation
        # Construct aggregation prompt
        aggregated_context = "\n\n".join(proposer_responses)

        aggregator_messages = [m.copy() for m in messages]

        # We inject the proposer responses into the context for the aggregator
        system_prompt = (
            "You are an aggregator model. You have received responses from multiple other models regarding the user's query. "
            "Synthesize these responses into a single, high-quality answer. "
            "Critically evaluate the information provided."
        )

        # Insert system prompt at start or append to user prompt
        # Let's insert as system message if possible, or prepend to first message
        aggregator_messages.insert(0, {"role": "system", "content": system_prompt})

        # Add the context
        last_user_msg_idx = -1
        for i, m in enumerate(aggregator_messages):
            if m["role"] == "user":
                last_user_msg_idx = i

        if last_user_msg_idx != -1:
            aggregator_messages[last_user_msg_idx]["content"] += (
                f"\n\nReference Responses from other models:\n{aggregated_context}"
            )
        else:
            # Fallback if no user message found (rare)
            aggregator_messages.append(
                {
                    "role": "user",
                    "content": f"Reference Responses:\n{aggregated_context}",
                }
            )

        print(f"  [MoA] Aggregating with {model_config}")
        start = time.time()
        final_response = self.simple_completion(
            model=model_config, messages=aggregator_messages, **litellm_params
        )
        duration = time.time() - start

        if trace_recorder:
            content = ""
            try:
                content = final_response.choices[0].message.content or ""
            except Exception:
                pass
            trace_recorder.add_node(
                step_type="aggregation",
                parent_id=strategy_node_id,
                model=model_config,
                role="assistant",
                content_preview=content[:200],
                duration_seconds=duration,
            )

        return final_response
