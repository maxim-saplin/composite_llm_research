# Composite LLM Research

This repository contains research and implementation for Composite LLM patterns (MoA, Council, Think, RLM) compatible with `litellm`.

See [DESIGN.md](DESIGN.md) for detailed architecture and usage.

## Example

```
══════════════════════════════════════════════════════════════════════
📊 SUMMARY
══════════════════════════════════════════════════════════════════════

Task: How many r's in strawberry?

  ✓ Llama-3.3-70b (Base)           → There are 2 R's and also 2 R's can be silent in the word "strawberry"
  ✓ Llama-3.1-8b (Base)            → There are 2 R's in the word "strawberry".
  ✓ CoT + Llama-70b                → There are 3 'r's in the word "strawberry".
  ✓ CoT + Llama-8b                 → Based on my counting, there are **3** Rs in the word "strawberry."
  ✓ ThinkTool + Llama-70b          → There are 2 r's in the word "strawberry" and also 2 r's are together in the word.
  ✓ MoA (Agg: 70b, Prop: [8b, Qwen]) → ...he conclusion that there are 3 "r"s in the word "strawberry". Therefore, the final answer is: **3**.
```

## Quick Start

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
python demo.py
```

## Council Strategy (LLM Council-style)

The `CouncilStrategy` adapts the 3-stage flow from [karpathy/llm-council](https://github.com/karpathy/llm-council):

1. First opinions from multiple council models
2. Cross-review and ranking of anonymized answers
3. Chairman model synthesizes the final answer

Example usage with `litellm`:

```python
from demo import composite_completion

resp = composite_completion(
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