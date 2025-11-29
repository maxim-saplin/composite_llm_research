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

from typing import List, Dict, Any
import time

from .base import BaseStrategy


DEFAULT_COUNCIL_MODELS = [
    # Defaults inspired by COUNCIL_MODELS in karpathy/llm-council
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
]


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

    def execute(
        self,
        messages: List[Dict[str, str]],
        model_config: str,
        optional_params: Dict[str, Any],
        litellm_params: Dict[str, Any],
        ) -> Any:
        # Configuration
        council_models = optional_params.get("council_models") or DEFAULT_COUNCIL_MODELS
        review_models = optional_params.get("review_models") or council_models
        chairman_model = optional_params.get("chairman_model") or model_config or (
            council_models[0] if council_models else "openai/gpt-4.1-mini"
        )
        max_council_size = optional_params.get("max_council_size")
        include_stage_summaries = optional_params.get(
            "include_stage_summaries_in_reasoning", True
        )

        if max_council_size is not None and max_council_size > 0:
            council_models = council_models[:max_council_size]
            review_models = review_models[:max_council_size]

        trace_recorder = optional_params.get("trace_recorder")
        trace_root_id = optional_params.get("trace_root_node_id")
        strategy_node_id = None
        if trace_recorder and trace_root_id is not None:
            strategy_node_id = trace_recorder.add_node(
                step_type="strategy",
                parent_id=trace_root_id,
                model=chairman_model,
                content_preview="Council strategy execution",
            )

        # --- Stage 1: First Opinions ---
        council_answers: List[Dict[str, str]] = []

        for idx, model_name in enumerate(council_models):
            alias = f"Assistant {idx + 1}"
            try:
                start = time.time()
                response = self.simple_completion(
                    model=model_name, messages=messages, **litellm_params
                )
                duration = time.time() - start
                content = response.choices[0].message.content
            except Exception as e:
                duration = None
                content = f"[ERROR from model '{model_name}': {e}]"

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
                    duration_seconds=duration,
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

        for i in range(num_reviewers):
            reviewer_model = review_models[i]
            reviewer_alias = f"Reviewer {i + 1}"

            # Construct review prompt: original conversation + council answers
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
                    **litellm_params,
                )
                duration = time.time() - start
                review_content = review_resp.choices[0].message.content
            except Exception as e:
                duration = None
                review_content = f"[ERROR from reviewer model '{reviewer_model}': {e}]"

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
                    duration_seconds=duration,
                    extra={"stage": "stage2", "reviewer_alias": reviewer_alias},
                )

        # --- Stage 3: Chairman Synthesis ---
        # Prepare summaries for the chairman
        stage1_summary_lines: List[str] = ["=== Stage 1: Council Answers ==="]
        for ans in council_answers:
            stage1_summary_lines.append(f"{ans['alias']}:")
            stage1_summary_lines.append(ans["content"])
            stage1_summary_lines.append("")
        stage1_summary = "\n".join(stage1_summary_lines).strip()

        stage2_summary_lines: List[str] = ["=== Stage 2: Cross-Reviews & Rankings ==="]
        for rev in reviews:
            stage2_summary_lines.append(f"{rev['reviewer_alias']} (model: {rev['model']}):")
            stage2_summary_lines.append(rev["content"])
            stage2_summary_lines.append("")
        stage2_summary = "\n".join(stage2_summary_lines).strip()

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
                    f"{stage1_summary}\n\n"
                    f"{stage2_summary}\n\n"
                    "Now, as the chairman, provide your final answer to the user. "
                    "Do not mention the council or the internal process explicitly "
                    "unless the user has asked you to."
                ),
            }
        )

        start = time.time()
        final_response = self.simple_completion(
            model=chairman_model,
            messages=chairman_messages,
            **litellm_params,
        )
        duration = time.time() - start

        if trace_recorder:
            content = ""
            try:
                content = final_response.choices[0].message.content or ""
            except Exception:
                pass
            trace_recorder.add_node(
                step_type="chairman",
                parent_id=strategy_node_id,
                model=chairman_model,
                role="assistant",
                content_preview=content[:200],
                duration_seconds=duration,
                extra={"stage": "stage3"},
            )

        # Attach a compact trace of the council process to reasoning_content
        if (
            hasattr(final_response, "choices")
            and final_response.choices
            and hasattr(final_response.choices[0], "message")
        ):
            msg = final_response.choices[0].message
            if include_stage_summaries:
                trace_parts = [
                    "LLM Council Trace:",
                    "",
                    stage1_summary,
                    "",
                    stage2_summary,
                ]
                trace_text = "\n".join(trace_parts).strip()
            else:
                trace_text = "LLM Council executed (stage summaries omitted)."

            # Attach as a non-standard field; many SDKs allow arbitrary attributes
            setattr(msg, "reasoning_content", trace_text)

        return final_response


