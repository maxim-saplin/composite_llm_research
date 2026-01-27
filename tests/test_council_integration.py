from typing import Any, cast
from pathlib import Path

import litellm

from composite_llm.litellm_provider import get_composite_provider


def _setup_composite_provider_from_config() -> None:
    repo = Path(__file__).resolve().parents[1]
    config_path = repo / "litellm_proxy.yaml"
    if not config_path.exists():
        config_path = repo / "litellm_proxy.example.yaml"
    config_file_path = str(config_path)

    provider = get_composite_provider(config_file_path=config_file_path)
    litellm.custom_provider_map = [{"provider": "composite", "custom_handler": provider}]
    from litellm.utils import custom_llm_setup

    custom_llm_setup()


def test_council_integration_returns_trace_and_usage() -> None:
    _setup_composite_provider_from_config()

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
