"""
Council Strategy - LLM Council-style 3-stage orchestration

Inspired by Andrej Karpathy's LLM Council project:
https://github.com/karpathy/llm-council

This strategy runs a 3-stage pipeline:

1. Stage 1 – First Opinions
   - Send the original conversation to a list of council models.
   - Collect their raw answers (one per model).

2. Stage 2 – Cross-review & Ranking
   - Each (review) model is shown ALL anonymized answers
     (labeled e.g. "Assistant 1", "Assistant 2", ...).
   - It critiques and ranks them by accuracy and insight.

3. Stage 3 – Chairman Synthesis
   - A designated chairman model receives:
       * The original conversation
       * A formatted summary of Stage 1 answers
       * A formatted summary of Stage 2 reviews
   - It produces the final answer for the user.

The client sees a single chat completion response, but we attach a
`reasoning_content` field to the final message containing a textual trace
of the council deliberations (answers + reviews).
"""

from typing import List, Dict, Any, cast, Callable, Sequence, Awaitable
import asyncio
import time

from .base import BaseStrategy
from ..trace import extract_usage_metrics


DEFAULT_COUNCIL_MODELS = [
    # Defaults inspired by COUNCIL_MODELS in karpathy/llm-council
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
]


def _compact_text(text: str, max_len: int = 160) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_len:
        return normalized
    return f"{normalized[: max_len - 3]}..."


class CouncilStrategy(BaseStrategy):
    """
    LLM Council-style strategy with 3 stages:

    Usage (via CompositeLLMProvider):
        model="composite/council/<chairman_model>"

    Optional params (via `optional_params`):
        - council_models: List[str]
            Models that act as council members in Stage 1 (and by default Stage 2).
        - chairman_model: str
            Model used in Stage 3 to synthesize the final answer.
            Defaults to the `<chairman_model>` part of the composite model string.
        - review_models: List[str]
            Optional list of models used to perform cross-review in Stage 2.
            If omitted, we reuse `council_models` as reviewers.
        - max_council_size: int
            Optional cap on the number of council members to use.
        - include_stage_summaries_in_reasoning: bool
            If True (default), include Stage 1 & 2 summaries in `reasoning_content`.
    """

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
        # Configuration
        def _normalize_node_list(raw: Any) -> List[Dict[str, Any]]:
            nodes: List[Dict[str, Any]] = []
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and isinstance(item.get("model"), str):
                        nodes.append(item)
                    elif isinstance(item, str):
                        nodes.append({"model": item})
            return nodes

        council_nodes = _normalize_node_list(
            optional_params.get("council")
            or optional_params.get("council_models")
            or DEFAULT_COUNCIL_MODELS
        )
        if not council_nodes:
            council_nodes = [{"model": model} for model in DEFAULT_COUNCIL_MODELS]

        review_nodes = _normalize_node_list(
            optional_params.get("reviewers") or optional_params.get("review_models")
        )
        if not review_nodes:
            review_nodes = list(council_nodes)

        chairman_node = optional_params.get("chairman")
        if isinstance(chairman_node, dict) and isinstance(chairman_node.get("model"), str):
            chairman_model = chairman_node["model"]
            chairman_params = {
                **litellm_params,
                **cast(Dict[str, Any], chairman_node.get("litellm_params") or {}),
            }
        else:
            chairman_model = (
                optional_params.get("chairman_model")
                or model_config
                or (council_nodes[0]["model"] if council_nodes else "openai/gpt-4.1-mini")
            )
            chairman_params = dict(litellm_params)

        council_models = [node["model"] for node in council_nodes]
        review_models = [node["model"] for node in review_nodes]
        max_council_size = optional_params.get("max_council_size")
        include_stage_summaries = optional_params.get(
            "include_stage_summaries_in_reasoning", True
        )

        if max_council_size is not None and max_council_size > 0:
            council_nodes = council_nodes[:max_council_size]
            review_nodes = review_nodes[:max_council_size]
            council_models = [node["model"] for node in council_nodes]
            review_models = [node["model"] for node in review_nodes]

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
                model=chairman_model,
                content_preview="Council strategy execution",
            )

        # Resume path: only run chairman stage with captured messages
        if resume_state:
            final_model = resume_state.get("model", chairman_model)
            final_messages = resume_state.get("messages") or messages
            reasoning_content = resume_state.get("reasoning_content")
            tool_trace = resume_state.get("tool_trace") or []
            if isinstance(final_stage, dict):
                final_stage.update(
                    {
                        "model": final_model,
                        "messages": [m.copy() for m in final_messages],
                        "reasoning_content": reasoning_content,
                        "stage": "stage3",
                    }
                )

            start = time.time()
            final_response = self.simple_completion(
                model=final_model, messages=final_messages, **chairman_params
            )
            duration = time.time() - start
            final_usage = extract_usage_metrics(final_response)

            if trace_recorder:
                content = ""
                final_choices = cast(Any, getattr(final_response, "choices", None))
                if final_choices:
                    content = final_choices[0].message.content or ""  # type: ignore[attr-defined]
                trace_recorder.add_node(
                    step_type="chairman",
                    parent_id=strategy_node_id,
                    model=final_model,
                    role="assistant",
                    content_preview=content[:200],
                    duration_seconds=duration,
                    prompt_tokens=final_usage["prompt_tokens"],
                    completion_tokens=final_usage["completion_tokens"],
                    total_tokens=final_usage["total_tokens"],
                    cost=final_usage["cost"],
                    extra={"stage": "stage3", "resume": True},
                )

            choices = cast(Any, getattr(final_response, "choices", None))
            if choices and hasattr(choices[0], "message") and reasoning_content:
                msg = choices[0].message  # type: ignore[attr-defined]
                if tool_trace:
                    reasoning_content = "\n".join(
                        [reasoning_content, "", "Tool Trace:"] + tool_trace
                    )
                setattr(msg, "reasoning_content", reasoning_content)

            return final_response

        # --- Stage 1: First Opinions ---
        council_answers: List[Dict[str, str]] = []

        def _call_council_sync(node: Dict[str, Any], alias: str) -> Dict[str, Any]:
            model_name = cast(str, node.get("model"))
            node_params = {
                **litellm_params,
                **cast(Dict[str, Any], node.get("litellm_params") or {}),
            }
            try:
                start = time.time()
                response = self.simple_completion(
                    model=model_name, messages=messages, **node_params
                )
                duration = time.time() - start
                response_choices = cast(Any, getattr(response, "choices", None))  # type: ignore[attr-defined]
                content = ""
                if response_choices:
                    content = response_choices[0].message.content or ""  # type: ignore[attr-defined]
                usage = extract_usage_metrics(response)
                return {
                    "alias": alias,
                    "model": model_name,
                    "content": content,
                    "duration": duration,
                    "usage": usage,
                    "error": None,
                }
            except Exception as e:
                return {
                    "alias": alias,
                    "model": model_name,
                    "content": f"[ERROR from model '{model_name}': {e}]",
                    "duration": None,
                    "usage": None,
                    "error": str(e),
                }

        async def _call_council_async(node: Dict[str, Any], alias: str) -> Dict[str, Any]:
            return await asyncio.to_thread(_call_council_sync, node, alias)

        council_results = self._run_async_tasks(
            lambda: [
                _call_council_async(node, f"Assistant {idx + 1}")
                for idx, node in enumerate(council_nodes)
            ],
            fallback=lambda: [
                _call_council_sync(node, f"Assistant {idx + 1}")
                for idx, node in enumerate(council_nodes)
            ],
        )

        for result in council_results:
            alias = result["alias"]
            model_name = result["model"]
            content = result["content"]
            usage = result["usage"]
            council_answers.append(
                {
                    "alias": alias,
                    "model": model_name,
                    "content": content or "",
                }
            )

            if trace_recorder:
                trace_recorder.add_node(
                    step_type="llm_call",
                    parent_id=strategy_node_id,
                    model=model_name,
                    role="assistant",
                    content_preview=(content or "")[:200],
                    duration_seconds=result["duration"],
                    prompt_tokens=usage["prompt_tokens"] if usage else None,
                    completion_tokens=usage["completion_tokens"] if usage else None,
                    total_tokens=usage["total_tokens"] if usage else None,
                    cost=usage["cost"] if usage else None,
                    extra={"stage": "stage1", "alias": alias},
                )

        # --- Stage 2: Cross-review & Ranking ---
        reviews: List[Dict[str, str]] = []

        # Build a shared text block of anonymized answers
        answers_block_lines: List[str] = []
        for ans in council_answers:
            answers_block_lines.append(f"{ans['alias']} (anonymous):")
            answers_block_lines.append(ans["content"])
            answers_block_lines.append("")  # blank line separator
        answers_block = "\n".join(answers_block_lines).strip()

        # Determine how many reviewers to use (mirror number of answers)
        num_reviewers = min(len(council_answers), len(review_models))

        def _call_reviewer_sync(
            reviewer_node: Dict[str, Any], reviewer_alias: str
        ) -> Dict[str, Any]:
            reviewer_model = cast(str, reviewer_node.get("model"))
            reviewer_params = {
                **litellm_params,
                **cast(Dict[str, Any], reviewer_node.get("litellm_params") or {}),
            }
            review_messages = [m.copy() for m in messages]
            review_messages.append(
                {
                    "role": "user",
                    "content": (
                        "You are part of an anonymous LLM council.\n\n"
                        "Below are several candidate answers to the user's request, "
                        "each labeled as Assistant 1, Assistant 2, etc. The true "
                        "model identities are hidden.\n\n"
                        "Your tasks:\n"
                        "1. Critique each assistant's answer for accuracy, completeness, and clarity.\n"
                        "2. Provide a ranking from best to worst assistant.\n"
                        "3. Explain briefly why the top-ranked answer is best.\n\n"
                        "Candidate answers:\n\n"
                        f"{answers_block}\n\n"
                        "Return your review and ranking in clear, readable text."
                    ),
                }
            )

            try:
                start = time.time()
                review_resp = self.simple_completion(
                    model=reviewer_model,
                    messages=review_messages,
                    **reviewer_params,
                )
                duration = time.time() - start
                review_choices = cast(Any, getattr(review_resp, "choices", None))  # type: ignore[attr-defined]
                review_content = ""
                if review_choices:
                    review_content = review_choices[0].message.content or ""  # type: ignore[attr-defined]
                review_usage = extract_usage_metrics(review_resp)
                return {
                    "reviewer_alias": reviewer_alias,
                    "model": reviewer_model,
                    "content": review_content,
                    "duration": duration,
                    "usage": review_usage,
                    "error": None,
                }
            except Exception as e:
                return {
                    "reviewer_alias": reviewer_alias,
                    "model": reviewer_model,
                    "content": f"[ERROR from reviewer model '{reviewer_model}': {e}]",
                    "duration": None,
                    "usage": None,
                    "error": str(e),
                }

        async def _call_reviewer_async(
            reviewer_node: Dict[str, Any], reviewer_alias: str
        ) -> Dict[str, Any]:
            return await asyncio.to_thread(
                _call_reviewer_sync, reviewer_node, reviewer_alias
            )

        review_results = self._run_async_tasks(
            lambda: [
                _call_reviewer_async(review_nodes[i], f"Reviewer {i + 1}")
                for i in range(num_reviewers)
            ],
            fallback=lambda: [
                _call_reviewer_sync(review_nodes[i], f"Reviewer {i + 1}")
                for i in range(num_reviewers)
            ],
        )

        for result in review_results:
            reviewer_alias = result["reviewer_alias"]
            reviewer_model = result["model"]
            review_content = result["content"]
            review_usage = result["usage"]

            reviews.append(
                {
                    "reviewer_alias": reviewer_alias,
                    "model": reviewer_model,
                    "content": review_content or "",
                }
            )

            if trace_recorder:
                trace_recorder.add_node(
                    step_type="review",
                    parent_id=strategy_node_id,
                    model=reviewer_model,
                    role="assistant",
                    content_preview=(review_content or "")[:200],
                    duration_seconds=result["duration"],
                    prompt_tokens=review_usage["prompt_tokens"]
                    if review_usage
                    else None,
                    completion_tokens=review_usage["completion_tokens"]
                    if review_usage
                    else None,
                    total_tokens=review_usage["total_tokens"] if review_usage else None,
                    cost=review_usage["cost"] if review_usage else None,
                    extra={"stage": "stage2", "reviewer_alias": reviewer_alias},
                )

        # --- Stage 3: Chairman Synthesis ---
        # Prepare summaries for the chairman
        stage1_summary_full_lines: List[str] = ["=== Stage 1: Council Answers ==="]
        for ans in council_answers:
            stage1_summary_full_lines.append(f"{ans['alias']}:")
            stage1_summary_full_lines.append(ans["content"])
            stage1_summary_full_lines.append("")
        stage1_summary_full = "\n".join(stage1_summary_full_lines).strip()

        stage2_summary_full_lines: List[str] = [
            "=== Stage 2: Cross-Reviews & Rankings ==="
        ]
        for rev in reviews:
            stage2_summary_full_lines.append(
                f"{rev['reviewer_alias']} (model: {rev['model']}):"
            )
            stage2_summary_full_lines.append(rev["content"])
            stage2_summary_full_lines.append("")
        stage2_summary_full = "\n".join(stage2_summary_full_lines).strip()

        stage1_summary_compact_lines: List[str] = ["Stage 1 summary:"]
        for ans in council_answers:
            compact = _compact_text(ans["content"])
            stage1_summary_compact_lines.append(
                f"- {ans['alias']} ({ans['model']}): {compact}"
            )
        stage1_summary_compact = "\n".join(stage1_summary_compact_lines).strip()

        stage2_summary_compact_lines: List[str] = ["Stage 2 summary:"]
        for rev in reviews:
            compact = _compact_text(rev["content"])
            stage2_summary_compact_lines.append(
                f"- {rev['reviewer_alias']} ({rev['model']}): {compact}"
            )
        stage2_summary_compact = "\n".join(stage2_summary_compact_lines).strip()

        chairman_messages = [m.copy() for m in messages]

        chairman_instructions = (
            "You are the chairman of an LLM council. Multiple anonymous assistants "
            "have proposed answers to the user's request, and multiple reviewers "
            "have critiqued and ranked those answers.\n\n"
            "Your job is to carefully read the council's answers and the reviewers' "
            "feedback, then produce a single, high-quality final answer.\n\n"
            "Guidelines:\n"
            "- Prefer answers that are factually correct, precise, and well-reasoned.\n"
            "- Fix any identified mistakes or gaps.\n"
            "- Be concise but thorough, and respond directly to the user's needs.\n\n"
            "Here is a summary of the council deliberation:"
        )

        chairman_messages.append(
            {
                "role": "user",
                "content": (
                    f"{chairman_instructions}\n\n"
                    f"{stage1_summary_full}\n\n"
                    f"{stage2_summary_full}\n\n"
                    "Now, as the chairman, provide your final answer to the user. "
                    "Do not mention the council or the internal process explicitly "
                    "unless the user has asked you to."
                ),
            }
        )

        reasoning_parts = [
            "LLM Council Trace:",
            f"Stage 1 models: {', '.join(council_models)}"
            if council_models
            else "Stage 1 models: none",
            f"Stage 2 reviewers: {', '.join(review_models[:num_reviewers])}"
            if num_reviewers
            else "Stage 2 reviewers: none",
            f"Stage 3 chairman: {chairman_model}",
            "",
            stage1_summary_compact
            if include_stage_summaries
            else "Stage 1 summary omitted.",
            "",
            stage2_summary_compact
            if include_stage_summaries
            else "Stage 2 summary omitted.",
            "",
            "Stage 3 summary:",
            f"- Chairman synthesized final response using {chairman_model}.",
        ]
        reasoning_content = "\n".join(reasoning_parts).strip()
        tool_trace: List[str] = []
        if tool_trace_context:
            tool_trace = list(tool_trace_context)
            reasoning_content = "\n".join(
                [reasoning_content, "", "Tool Trace:"] + tool_trace
            )
        if isinstance(final_stage, dict):
            final_stage.update(
                {
                    "model": chairman_model,
                    "messages": [m.copy() for m in chairman_messages],
                    "reasoning_content": reasoning_content,
                    "tool_trace": tool_trace,
                    "stage": "stage3",
                }
            )

        start = time.time()
        final_response = self.simple_completion(
            model=chairman_model,
            messages=chairman_messages,
            **chairman_params,
        )
        duration = time.time() - start
        final_usage = extract_usage_metrics(final_response)

        if trace_recorder:
            content = ""
            final_choices = cast(Any, getattr(final_response, "choices", None))  # type: ignore[attr-defined]
            if final_choices:
                content = final_choices[0].message.content or ""  # type: ignore[attr-defined]
            trace_recorder.add_node(
                step_type="chairman",
                parent_id=strategy_node_id,
                model=chairman_model,
                role="assistant",
                content_preview=content[:200],
                duration_seconds=duration,
                prompt_tokens=final_usage["prompt_tokens"],
                completion_tokens=final_usage["completion_tokens"],
                total_tokens=final_usage["total_tokens"],
                cost=final_usage["cost"],
                extra={"stage": "stage3"},
            )

        # Attach a compact trace of the council process to reasoning_content
        choices = cast(Any, getattr(final_response, "choices", None))
        if choices and hasattr(choices[0], "message"):
            msg = choices[0].message  # type: ignore[attr-defined]
            # Attach as a non-standard field; many SDKs allow arbitrary attributes
            setattr(msg, "reasoning_content", reasoning_content)

        return final_response
