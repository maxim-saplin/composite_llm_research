from __future__ import annotations

from typing import Dict, List

CEREBRAS_MODELS: Dict[str, str] = {
    "llama-70b": "cerebras/llama-3.3-70b",
    "llama-8b": "cerebras/llama3.1-8b",
    "qwen-32b": "cerebras/qwen-3-32b",
}


def get_moa_model() -> str:
    return f"composite/moa/{CEREBRAS_MODELS['llama-70b']}"


def get_moa_optional_params() -> Dict[str, List[str]]:
    return {
        "proposers": [
            CEREBRAS_MODELS["llama-8b"],
            CEREBRAS_MODELS["qwen-32b"],
        ]
    }


def get_council_params() -> Dict[str, List[str] | str]:
    return {
        "chairman_model": CEREBRAS_MODELS["llama-70b"],
        "council_models": [
            CEREBRAS_MODELS["llama-8b"],
            CEREBRAS_MODELS["qwen-32b"],
        ],
    }