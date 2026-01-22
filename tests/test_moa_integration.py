from typing import Any, cast

import litellm

from composite_llm.litellm_provider import register_composite_provider


def test_moa_integration_returns_trace_and_usage() -> None:
    register_composite_provider()

    response = cast(Any, litellm.completion(
        model="composite/moa/openai/aggregator-model",
        messages=[{"role": "user", "content": "Summarize the task."}],
        optional_params={
            "proposers": ["openai/proposer-1", "openai/proposer-2"],
        },
    ))

    message = response.choices[0].message
    content = str(getattr(message, "content", "") or "")
    assert "Aggregated answer" in content
    assert hasattr(message, "reasoning_content")
    assert "Proposers" in message.reasoning_content
    usage = cast(Any, getattr(response, "usage", {}))
    total_tokens = usage["total_tokens"] if isinstance(usage, dict) else usage.total_tokens
    assert total_tokens > 0
