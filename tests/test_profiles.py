from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import litellm
import yaml

from composite_llm.litellm_provider import get_composite_provider


def _setup_provider(config_path: Path) -> None:
    provider = get_composite_provider(config_file_path=str(config_path))
    litellm.custom_provider_map = [{"provider": "composite", "custom_handler": provider}]
    from litellm.utils import custom_llm_setup

    custom_llm_setup()


def _write_profile_config(tmp_path: Path) -> Path:
    config = {
        "general_settings": {
            "composite_llm": {
                "version": 1,
                "providers": {
                    "openai_test": {
                        "litellm_params": {
                            "api_base": "os.environ/OPENAI_API_BASE",
                            "api_key": "os.environ/OPENAI_API_KEY",
                        }
                    }
                },
                "profiles": {
                    "moa_light": {
                        "strategy": "moa",
                        "topology": {
                            "aggregator": {
                                "model": "openai/aggregator-model",
                                "provider": "openai_test",
                            },
                            "proposers": [
                                {
                                    "model": "openai/proposer-1",
                                    "provider": "openai_test",
                                },
                                {
                                    "model": "openai/proposer-2",
                                    "provider": "openai_test",
                                },
                            ],
                            "settings": {"max_tool_iterations": 2},
                        },
                    }
                },
            }
        }
    }

    path = tmp_path / "litellm_proxy.profile.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_profile_model_resolves(fake_openai_server: Any, tmp_path: Path) -> None:
    config_path = _write_profile_config(tmp_path)
    _setup_provider(config_path)

    response = cast(
        Any,
        litellm.completion(
            model="composite/profile/moa_light",
            messages=[{"role": "user", "content": "Summarize the task."}],
        ),
    )

    message = response.choices[0].message
    content = str(getattr(message, "content", "") or "")
    assert "Aggregated answer" in content
    assert hasattr(message, "reasoning_content")
    reasoning = str(getattr(message, "reasoning_content", "") or "")
    assert "Proposer summaries" in reasoning


def test_profile_override_allowed(fake_openai_server: Any, tmp_path: Path) -> None:
    config_path = _write_profile_config(tmp_path)
    _setup_provider(config_path)

    response = cast(
        Any,
        litellm.completion(
            model="composite/profile/moa_light",
            messages=[{"role": "user", "content": "Summarize the task."}],
            optional_params={"proposers": ["openai/proposer-1"]},
        ),
    )

    message = response.choices[0].message
    content = str(getattr(message, "content", "") or "")
    assert "Aggregated answer" in content
