from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from composite_llm.proxy_bootstrap import bootstrap


def _default_config_path() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    primary = repo_root / "litellm_proxy.yaml"
    if primary.exists():
        return str(primary)
    example = repo_root / "litellm_proxy.example.yaml"
    return str(example)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LiteLLM proxy with composite provider")
    parser.add_argument(
        "--config",
        default=_default_config_path(),
        help="Path to LiteLLM proxy YAML config",
    )
    args = parser.parse_args()

    bootstrap()

    # LiteLLM proxy reads config path from env when running via uvicorn.
    # Only set if the user hasn't provided one already.
    os.environ.setdefault("LITELLM_CONFIG", str(args.config))

    from litellm.proxy.proxy_server import app

    uvicorn.run(app, host="0.0.0.0", port=4000)


if __name__ == "__main__":
    main()