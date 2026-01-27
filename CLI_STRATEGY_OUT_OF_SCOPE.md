# CLI-Driven Strategy / Non-LiteLLM Inference (Out of Scope)

## Status
Out of scope for the current plan in [PLAN.md](PLAN.md). This document captures requirements and design considerations so we can tackle it next without muddying the LiteLLM-aligned config work.

---

## Problem statement
We want to support strategies (or internal nodes within strategies) that do **not** call `litellm.completion`.

Example: a local CLI that performs inference/state management internally.

```bash
agent -p "Hi! What's your name" --model "composer-1" --mode=ask
```

In this world, each “turn” may require passing the entire USER/ASSISTANT transcript (and potentially tool results) as plain text into the CLI.

---

## Why this is meaningfully different from LiteLLM-backed nodes
LiteLLM-backed nodes already provide:
- A standard request schema (messages, tools, etc.)
- A consistent response schema (choices/message/tool_calls)
- Provider configuration and retry/timeout behavior

A CLI-driven backend introduces:
- **State serialization**: transcript format, delimiters, role markers, and escaping rules
- **State persistence**: where state lives between turns (in-memory vs temp files vs explicit state tokens)
- **Tool integration**: how tool calls are represented and round-tripped
- **Error surfaces**: exit codes, stderr parsing, partial output
- **Security**: safe subprocess invocation (no shell), prompt injection into flags, path resolution
- **Performance**: process startup cost, concurrency, streaming vs non-streaming

---

## Requirements (proposed)

### R1: Safe execution
- Use `subprocess.run([...], shell=False)` (no shell interpolation)
- Explicit timeout support
- Capture and size-limit stdout/stderr

### R2: Deterministic protocol
Decide one of:
- **Plain text protocol** (simpler): CLI returns final assistant content only
- **JSON protocol** (recommended long-term): CLI returns a JSON object that can encode tool calls, intermediate steps, etc.

### R3: Conversation format
If plain text:
- Standardize how messages are serialized (e.g., `ROLE: content` blocks)
- Define escaping rules for `ROLE:` tokens, code blocks, etc.

If JSON:
- CLI accepts a JSON payload including `messages` and returns OpenAI-like JSON

### R4: Proxy friendliness
- Must be invokable from both Python and LiteLLM proxy
- Must fit within `model_list` aliases and/or `composite/profile/<name>` topologies

### R5: Observability
- Capture timing, exit code, stderr summaries
- Emit trace nodes in the same trace structure as LiteLLM-backed steps

---

## Design options

### Option A: Implement as a LiteLLM custom provider
Create a provider like `agentcli/<model>`.

Pros:
- Composable: can be used as a node inside MoA/Council topologies
- Works in Python and proxy the same way

Cons:
- Need to adapt CLI IO to OpenAI-like response objects

### Option B: Implement as a composite strategy backend (not a provider)
Create a strategy that calls the CLI directly.

Pros:
- Doesn’t need to pretend to be a standard LLM provider

Cons:
- Harder to compose with other nodes (e.g., MoA with one CLI proposer)

---

## Recommended next step (when in scope)
1) Pick protocol (plain text vs JSON). If uncertain, start with plain text but design for JSON upgrade.
2) Implement a minimal `agentcli` LiteLLM custom provider behind a feature flag.
3) Add a test stub CLI script (deterministic output) to validate:
   - multi-turn transcript passing
   - timeout handling
   - stderr handling
4) Document the protocol and configuration knobs.
