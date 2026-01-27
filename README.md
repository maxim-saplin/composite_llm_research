# Composite LLM Research

This repository contains a working implementation of Composite LLM patterns (MoA, Council, Composer CLI) compatible with LiteLLM.

For the detailed technical layout, see [TECH.md](TECH.md).

## Example

```
══════════════════════════════════════════════════════════════════════
📊 SUMMARY
══════════════════════════════════════════════════════════════════════

Task: How many r's in strawberry?

  ✓ Llama-3.3-70b (Base)           → There are 2 R's and also 2 R's can be silent in the word "strawberry"
  ✓ Llama-3.1-8b (Base)            → There are 2 R's in the word "strawberry".
  ✓ MoA (Agg: 70b, Prop: [8b, Qwen]) → ...he conclusion that there are 3 "r"s in the word "strawberry". Therefore, the final answer is: **3**.
```

## High-level overview

- Composite strategies are implemented in Python and exposed as a LiteLLM custom provider.
- The proxy exposes public model names via `model_list` and maps them to composite profiles.
- Profiles define the full topology (nodes + settings) in YAML.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Set API keys (either works for the demo):

- `CEREBRAS_API_KEY` for direct Cerebras calls.
- `LITELLM_API_KEY` for LiteLLM proxy (demo maps this to Cerebras automatically).

Composite strategy model string format: `composite/<strategy>/<provider>/<model>`.
Example: `composite/council/cerebras/llama-3.3-70b`.

Composer CLI model string format: `composite/composer-cli/<model>`.
Example: `composite/composer-cli/composer-1`.

Profile-based model string format (recommended for proxy): `composite/profile/<name>`.
Example: `composite/profile/moa_light`.

## Configs

- Example proxy config: [litellm_proxy.example.yaml](litellm_proxy.example.yaml)
- Sample env file: [sample.env](sample.env)
- Proxy launcher (sources .env): [scripts/run_proxy_from_env.sh](scripts/run_proxy_from_env.sh)

## Proxy config

Use the checked-in example config as a starting point, then copy it locally:

```bash
cp litellm_proxy.example.yaml litellm_proxy.yaml
```

See [PROXY_USAGE.md](PROXY_USAGE.md) for full proxy instructions.

## Troubleshooting

- LiteLLM Provider List warnings: set `litellm.suppress_debug_info = True` before making requests to silence them.
- Auth errors: ensure `CEREBRAS_API_KEY` or `LITELLM_API_KEY` is set in your environment.

## Council Strategy (LLM Council-style)

The `CouncilStrategy` adapts the 3-stage flow from [karpathy/llm-council](https://github.com/karpathy/llm-council):

1. First opinions from multiple council models
2. Cross-review and ranking of anonymized answers
3. Chairman model synthesizes the final answer

Example usage with `litellm`:

```python
import litellm
from composite_llm.litellm_provider import register_composite_provider

register_composite_provider()

resp = litellm.completion(
    model="composite/council/cerebras/llama-3.3-70b",
    messages=[{"role": "user", "content": "How many r's in strawberry?"}],
    optional_params={
        "council_models": [
            "cerebras/llama3.1-8b",
            "cerebras/qwen-3-32b",
        ],
        "chairman_model": "cerebras/llama-3.3-70b",
    },
)
print(resp.choices[0].message.content)
```

## Demo scripts

- Plain demo: [demo.py](demo.py)
- Proxy demo: [scripts/demo_proxy.py](scripts/demo_proxy.py)
  - First run the proxy via chmod +x `scripts/run_proxy_from_env.sh` and `./scripts/run_proxy_from_env.sh`
- Proxy + Composer CLI test: [scripts/test_proxy_composer_cli.py](scripts/test_proxy_composer_cli.py)

## Demo + Tests

Run the demo:

```bash
python demo.py
```

Run the profile-based demo:

```bash
python demo_profiles.py
```

Run the proxy demo (pretty output):

```bash
litellm --config litellm_proxy.example.yaml --host 0.0.0.0 --port 4000
python scripts/demo_proxy.py --base-url http://localhost:4000
```

Run the proxy + Composer CLI test (requires `agent` on PATH):

```bash
python scripts/test_proxy_composer_cli.py
```

Run the integration tests (includes a fake OpenAI-compatible server):

```bash
uv pip install -e ".[test]"
uv run pytest -q
```

Re-run tests in the same `.venv` without recreating it.

Dashboard dependencies live in the extra: `uv pip install -e ".[dashboard]"`.
