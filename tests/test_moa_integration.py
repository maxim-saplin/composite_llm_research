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

    # Mirror LiteLLM proxy's YAML custom_provider_map behavior.
    provider = get_composite_provider(config_file_path=config_file_path)
    litellm.custom_provider_map = [{"provider": "composite", "custom_handler": provider}]
    from litellm.utils import custom_llm_setup

    custom_llm_setup()


def test_moa_integration_returns_trace_and_usage() -> None:
    _setup_composite_provider_from_config()

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
