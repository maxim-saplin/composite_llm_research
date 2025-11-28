from typing import List, Dict, Any
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

        # Step 1: Parallel calls to proposers
        # For simplicity in this synchronous demo, we loop. In production, use asyncio.gather
        proposer_responses = []

        print(f"  [MoA] Querying proposers: {proposers}")

        for p_model in proposers:
            try:
                # We reuse the same messages for proposers
                # Note: We should handle API keys for different providers in real app
                resp = self.simple_completion(
                    model=p_model, messages=messages, **litellm_params
                )
                content = resp.choices[0].message.content
                proposer_responses.append(f"Model {p_model} suggests:\n{content}")
            except Exception as e:
                proposer_responses.append(f"Model {p_model} failed: {str(e)}")

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
        final_response = self.simple_completion(
            model=model_config, messages=aggregator_messages, **litellm_params
        )

        return final_response
