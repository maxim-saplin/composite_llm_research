# Composite LLM Research

This repository contains research and implementation for Composite LLM patterns (MoA, Council) compatible with `litellm`.

See [DESIGN.md](DESIGN.md) for detailed architecture and usage.

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

## Setup

Requires Python >=3.13. `uv venv` uses the most recent installed Python unless you pass `--python` and reuses an existing `.venv`.

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

## Demo + Tests

Run the demo:

```bash
python demo.py
```

Run the integration tests (includes a fake OpenAI-compatible server):

```bash
uv pip install -e ".[test]"
uv run pytest -q
```

Re-run tests in the same `.venv` without recreating it.

Dashboard dependencies live in the extra: `uv pip install -e ".[dashboard]"`.
