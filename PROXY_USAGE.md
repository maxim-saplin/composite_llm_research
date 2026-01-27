# Proxy usage (YAML-first)

This repo’s composite provider is intended to be loaded by the LiteLLM proxy config (YAML), not by Python-side bootstrap code.

## Config file

Copy the example config and customize it for your providers/models:

```bash
cp litellm_proxy.example.yaml litellm_proxy.yaml
```

## Run locally

Option A (recommended): run LiteLLM CLI directly

- `litellm --config litellm_proxy.example.yaml --host 0.0.0.0 --port 4000`

Option B: use the repo’s small runner

- `python -m composite_llm.proxy_local --config litellm_proxy.yaml`

Option C: run via uvicorn

- `CONFIG_FILE_PATH=litellm_proxy.yaml uvicorn litellm.proxy.proxy_server:app --host 0.0.0.0 --port 4000`

## How the custom provider is loaded

The proxy config should include (LiteLLM >= 1.80 expects this under `litellm_settings`):

- `litellm_settings:
    custom_provider_map:
      - provider: composite
        custom_handler: composite_llm.litellm_provider.COMPOSITE_PROVIDER`

LiteLLM will import the handler object. It resolves the config path from the CLI `--config`, `CONFIG_FILE_PATH`, or `LITELLM_CONFIG`.

## Profiles (recommended)

The example config uses profile-backed models like `moa_light` and `council_basic`:

- `model_list` exposes public proxy names (e.g., `moa_light`)
- Each public name maps to `composite/profile/<profile_name>`
- Profile topology lives under `general_settings.composite_llm.profiles`

Default example models exposed by the config:

- `moa_light`
- `moa_hard`
- `council_basic`

## Quick validation & smoke test

- Validate your YAML has the required custom provider wiring:
  - `python scripts/validate_proxy_config.py --config litellm_proxy.yaml`
- With the proxy running, send a real request:
  - `python scripts/smoke_proxy_chat.py --base-url http://localhost:4000 --model moa_light`
- Or run the pretty demo (shows traces by default; use `--no-reasoning` to hide):
  - `python scripts/demo_proxy_profiles.py --base-url http://localhost:4000`

## Strategy registry (YAML)

To avoid hardcoded `if/else` strategy imports, strategies are resolved from YAML:

- `general_settings:
    composite_llm:
      strategy_registry:
        moa: composite_llm.strategies.moa:MoAStrategy
        council: composite_llm.strategies.council:CouncilStrategy`

Add a new strategy by implementing `BaseStrategy` and adding an import path entry.

## Strategy defaults

This config is explicit only (no global defaults). Put all topology and settings directly in each profile.

## Secrets

Keep secrets in env vars only. The example config uses `os.environ/...` for API keys.

## Observability

Logging to llm_logs.jsonl is disabled by default. To enable it, add to your composite config:

- `general_settings:
    composite_llm:
      observability:
        enabled: true
        log_file: llm_logs.jsonl`
