from typing import List, Dict, Any, cast, Callable, Sequence, Awaitable, Optional
import asyncio
import time

import litellm

from .base import BaseStrategy
from ..trace import extract_usage_metrics


def _compact_text(text: str, max_len: int = 160) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_len:
        return normalized
    return f"{normalized[: max_len - 3]}..."


class MoAStrategy(BaseStrategy):
    def _run_async_tasks(
        self,
        build_coroutines: Callable[[], Sequence[Awaitable[Dict[str, Any]]]],
        fallback: Callable[[], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return fallback()

        async def _runner() -> List[Dict[str, Any]]:
            return await asyncio.gather(*build_coroutines())

        return asyncio.run(_runner())

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

        raw_proposers = optional_params.get(
            "proposers", ["cerebras/llama3.1-8b", "cerebras/qwen-3-32b"]
        )

        proposer_nodes: List[Dict[str, Any]] = []
        if isinstance(raw_proposers, list):
            for item in raw_proposers:
                if isinstance(item, dict) and isinstance(item.get("model"), str):
                    proposer_nodes.append(item)
                elif isinstance(item, str):
                    proposer_nodes.append({"model": item})

        if not proposer_nodes:
            proposer_nodes = [
                {"model": "cerebras/llama3.1-8b"},
                {"model": "cerebras/qwen-3-32b"},
            ]

        proposer_models = [node["model"] for node in proposer_nodes]

        aggregator_node = optional_params.get("aggregator")
        if isinstance(aggregator_node, dict) and isinstance(
            aggregator_node.get("model"), str
        ):
            aggregator_model = aggregator_node["model"]
            aggregator_params = {
                **litellm_params,
                **cast(Dict[str, Any], aggregator_node.get("litellm_params") or {}),
            }
        else:
            aggregator_model = model_config
            aggregator_params = dict(litellm_params)

        trace_recorder = optional_params.get("trace_recorder")
        trace_root_id = optional_params.get("trace_root_node_id")
        resume_state = optional_params.get("resume_state")
        final_stage = optional_params.get("final_stage")
        tool_trace_context = optional_params.get("tool_trace_context")

        strategy_node_id = None
        if trace_recorder and trace_root_id is not None:
            strategy_node_id = trace_recorder.add_node(
                step_type="strategy",
                parent_id=trace_root_id,
                model=aggregator_model,
                content_preview="MoA strategy execution",
            )

        if resume_state:
            final_model = resume_state.get("model", aggregator_model)
            final_messages = resume_state.get("messages") or messages
            reasoning_content = resume_state.get("reasoning_content")
            tool_trace = resume_state.get("tool_trace") or []
            if isinstance(final_stage, dict):
                final_stage.update(
                    {
                        "model": final_model,
                        "messages": [m.copy() for m in final_messages],
                        "reasoning_content": reasoning_content,
                        "stage": "aggregation",
                    }
                )

            start = time.time()
            final_response = self.simple_completion(
                model=final_model, messages=final_messages, **aggregator_params
            )
            duration = time.time() - start
            usage = extract_usage_metrics(final_response)

            if trace_recorder:
                content = ""
                try:
                    content = final_response.choices[0].message.content or ""  # type: ignore[attr-defined]
                except Exception:
                    pass
                trace_recorder.add_node(
                    step_type="aggregation",
                    parent_id=strategy_node_id,
                    model=final_model,
                    role="assistant",
                    content_preview=content[:200],
                    duration_seconds=duration,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                    cost=usage["cost"],
                    extra={"resume": True},
                )

            final_choices = cast(Any, getattr(final_response, "choices", None))
            if final_choices and hasattr(final_choices[0], "message"):
                msg = final_choices[0].message  # type: ignore[attr-defined]
                if reasoning_content:
                    if tool_trace:
                        reasoning_content = "\n".join(
                            [reasoning_content, "", "Tool Trace:"] + tool_trace
                        )
                    setattr(msg, "reasoning_content", reasoning_content)

            return final_response

        # Step 1: Parallel calls to proposers
        proposer_responses: List[str] = []

        print(f"  [MoA] Querying proposers: {proposer_models}")

        def _call_proposer_sync(node: Dict[str, Any]) -> Dict[str, Any]:
            p_model = cast(str, node.get("model"))
            node_params = {
                **litellm_params,
                **cast(Dict[str, Any], node.get("litellm_params") or {}),
            }
            try:
                start = time.time()
                resp = self.simple_completion(
                    model=p_model, messages=messages, **node_params
                )
                duration = time.time() - start
                resp_choices = cast(Any, getattr(resp, "choices", None))
                content = ""
                if resp_choices:
                    content = resp_choices[0].message.content or ""  # type: ignore[attr-defined]
                usage = extract_usage_metrics(resp)
                return {
                    "model": p_model,
                    "content": content,
                    "duration": duration,
                    "usage": usage,
                    "error": None,
                }
            except Exception as e:
                return {
                    "model": p_model,
                    "content": "",
                    "duration": None,
                    "usage": None,
                    "error": str(e),
                }

        async def _call_proposer_async(node: Dict[str, Any]) -> Dict[str, Any]:
            return await asyncio.to_thread(_call_proposer_sync, node)

        proposer_results = self._run_async_tasks(
            lambda: [_call_proposer_async(node) for node in proposer_nodes],
            fallback=lambda: [_call_proposer_sync(node) for node in proposer_nodes],
        )

        for result in proposer_results:
            model_name = result["model"]
            if result["error"]:
                error_msg = f"Model {model_name} failed: {result['error']}"
                proposer_responses.append(error_msg)
                if trace_recorder:
                    trace_recorder.add_node(
                        step_type="llm_call",
                        parent_id=strategy_node_id,
                        model=model_name,
                        role="assistant",
                        content_preview=error_msg[:200],
                    )
                continue

            content = result["content"]
            usage = result["usage"]
            proposer_responses.append(f"Model {model_name} suggests:\n{content}")

            if trace_recorder and usage:
                trace_recorder.add_node(
                    step_type="llm_call",
                    parent_id=strategy_node_id,
                    model=model_name,
                    role="assistant",
                    content_preview=(content or "")[:200],
                    duration_seconds=result["duration"],
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                    cost=usage["cost"],
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

        print(f"  [MoA] Aggregating with {aggregator_model}")
        proposer_summary_lines: List[str] = ["Proposer summaries:"]
        if proposer_results:
            for result in proposer_results:
                model_name = result["model"]
                if result.get("error"):
                    summary = f"ERROR: {result['error']}"
                else:
                    summary = _compact_text(result.get("content") or "")
                    if not summary:
                        summary = "(empty response)"
                proposer_summary_lines.append(f"- {model_name}: {summary}")
        else:
            proposer_summary_lines.append("- none")

        reasoning_content = "\n".join(
            [
                "MoA Trace:",
                f"Proposers: {', '.join(proposer_models)}"
                if proposer_models
                else "Proposers: none",
                f"Aggregator: {aggregator_model}",
                "",
                "\n".join(proposer_summary_lines).strip(),
            ]
        )
        tool_trace: List[str] = []
        if tool_trace_context:
            tool_trace = list(tool_trace_context)
            reasoning_content = "\n".join(
                [reasoning_content, "", "Tool Trace:"] + tool_trace
            )
        if isinstance(final_stage, dict):
            final_stage.update(
                {
                    "model": aggregator_model,
                    "messages": [m.copy() for m in aggregator_messages],
                    "reasoning_content": reasoning_content,
                    "tool_trace": tool_trace,
                    "stage": "aggregation",
                }
            )

        start = time.time()
        final_response = self.simple_completion(
            model=aggregator_model,
            messages=aggregator_messages,
            **aggregator_params,
        )
        duration = time.time() - start
        usage = extract_usage_metrics(final_response)

        if trace_recorder:
            content = ""
            try:
                content = final_response.choices[0].message.content or ""  # type: ignore[attr-defined]
            except Exception:
                pass
            trace_recorder.add_node(
                step_type="aggregation",
                parent_id=strategy_node_id,
                model=aggregator_model,
                role="assistant",
                content_preview=content[:200],
                duration_seconds=duration,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                cost=usage["cost"],
            )

        final_choices = cast(Any, getattr(final_response, "choices", None))
        if final_choices and hasattr(final_choices[0], "message"):
            msg = final_choices[0].message  # type: ignore[attr-defined]
            setattr(msg, "reasoning_content", reasoning_content)

        return final_response
