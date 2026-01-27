# Composite LLM Technical Overview

## Architecture

The system is a LiteLLM custom provider that orchestrates multi‑model strategies (MoA, Council). It can run:

- Direct Python calls (via `litellm.completion()` with composite model strings).
- LiteLLM proxy calls (via `model_list` + custom provider registration in YAML).

Key modules:

- [composite_llm/litellm_provider.py](composite_llm/litellm_provider.py): provider entrypoint, model parsing, profile resolution, and request execution.
- [composite_llm/strategies/moa.py](composite_llm/strategies/moa.py): Mixture‑of‑Agents strategy.
- [composite_llm/strategies/council.py](composite_llm/strategies/council.py): Council strategy.
- [composite_llm/config_schema.py](composite_llm/config_schema.py): YAML schema validation for profiles and provider bundles.
- [composite_llm/observability.py](composite_llm/observability.py): optional file logging.

## Model routing

Supported model strings:

- `composite/<strategy>/<provider>/<model>` (legacy, direct calls)
- `composite/profile/<name>` (recommended for proxy)

When using profiles, the provider resolves the profile topology from YAML and materializes per‑node LiteLLM params, then executes the strategy.

## YAML layout (LiteLLM proxy)

### Custom provider wiring

In the proxy config:

- `litellm_settings.custom_provider_map` registers the composite handler.
- `model_list` exposes public names and maps them to composite profile model strings.

Example is in [litellm_proxy.example.yaml](litellm_proxy.example.yaml).

### Composite settings

Under `general_settings.composite_llm`:

- `strategy_registry`: maps strategy names to Python classes.
- `providers`: reusable LiteLLM param bundles (e.g., keys, api_base).
- `profiles`: full topology and settings per profile.

Profiles are explicit (no implicit defaults). Each profile defines its full node topology.

## Profiles and nodes

A profile defines:

- `strategy`: `moa` or `council`.
- `topology`: strategy‑specific node lists and `settings`.

Nodes look like:

- `model`: model string for LiteLLM (e.g., `cerebras/llama-3.3-70b`).
- `provider`: optional provider bundle name.
- `params`: per‑node overrides (merged on top of provider bundle).

The provider resolves provider bundles and node params and passes them to the strategy.

## Strategy execution

### MoA

- Proposers run in parallel.
- Aggregator merges proposer outputs and returns a final answer.
- Trace is attached to `reasoning_content` with proposer summaries.

### Council

- Stage 1: council member answers.
- Stage 2: reviewers critique and rank.
- Stage 3: chairman synthesizes.
- Trace is attached to `reasoning_content` with compact stage summaries.

## Overrides and params

- Requests may include `optional_params` to override settings.
- Profiles supply explicit `settings` and node params.
- Merge order: profile settings → request `optional_params`.

## Observability (optional)

File logging is disabled by default. Enable in config:

```
general_settings:
  composite_llm:
    observability:
      enabled: true
      log_file: llm_logs.jsonl
```

See [composite_llm/observability.py](composite_llm/observability.py).

## Proxy execution flow

1. LiteLLM proxy loads config and registers `COMPOSITE_PROVIDER`.
2. Proxy routes public model names from `model_list`.
3. Provider resolves profile topology and executes the strategy.
4. Responses include `reasoning_content` traces.

## Demos

- Plain demo: [demo.py](demo.py)
- Profile demo: [demo_profiles.py](demo_profiles.py)
- Proxy demo: [scripts/demo_proxy.py](scripts/demo_proxy.py)

## Tests

Run:

- `uv pip install -e ".[test]"`
- `uv run pytest -q`

## Known constraints

- Only `moa` and `council` are implemented.
- No implicit strategy defaults; profiles must be explicit.
- Profile validation is schema‑based and rejects unknown provider references.
