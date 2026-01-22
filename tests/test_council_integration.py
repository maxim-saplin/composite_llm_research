from typing import Any, cast

import litellm

from composite_llm.litellm_provider import register_composite_provider


def test_council_integration_returns_trace_and_usage() -> None:
    register_composite_provider()

    response = cast(Any, litellm.completion(
        model="composite/council/openai/chairman-model",
        messages=[{"role": "user", "content": "Give a short update."}],
        optional_params={
            "council_models": ["openai/proposer-1", "openai/proposer-2"],
            "review_models": ["openai/reviewer-1", "openai/reviewer-2"],
            "chairman_model": "openai/chairman-model",
        },
    ))

    message = response.choices[0].message
    content = str(getattr(message, "content", "") or "")
    assert "Chairman answer" in content
    assert hasattr(message, "reasoning_content")
    reasoning = str(getattr(message, "reasoning_content", "") or "")
    assert "LLM Council Trace" in reasoning
    usage = cast(Any, getattr(response, "usage", {}))
    total_tokens = usage["total_tokens"] if isinstance(usage, dict) else usage.total_tokens
    assert total_tokens > 0
