# CLI-Driven Strategy (Cursor `agent` CLI)

---

## Problem statement
We want to support strategies (or internal nodes within strategies) that do **not** call `litellm.completion`.

For the Cursor CLI, we will invoke the `agent` binary directly and treat its output as the final assistant content.

```bash
agent -p "Hi! What's your name" --model "composer-1" --mode=ask
```

Each “turn” will pass the full USER/ASSISTANT transcript as plain text into the CLI.

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

## Requirements

### R1: Safe execution
- Use `subprocess.run([...], shell=False)` (no shell interpolation)
- Explicit timeout support
- Capture and size-limit stdout/stderr

### R2: Deterministic protocol
- **Plain text protocol**: CLI returns final assistant content only
- No tool calls (text-only output)

### R3: Conversation format
Plain text transcript, reconstructed from chat completion messages.

- Preserve all messages in order
- Use explicit role delimiters unlikely to appear in content:
	- `<<<ROLE:SYSTEM>>>` (if present)
	- `<<<ROLE:USER>>>`
	- `<<<ROLE:ASSISTANT>>>`
- Insert a blank line between messages
- If content can include these exact tokens, escape by prefixing a backslash (runtime can handle this if needed)

### R4: Proxy friendliness
- Must be invokable from both Python and LiteLLM proxy
- Must fit within `model_list` aliases and/or `composite/profile/<name>` topologies

### R5: Observability
- Capture timing, exit code, stderr summaries
- Emit trace nodes in the same trace structure as LiteLLM-backed steps

### R6: Statelessness
- No persistence between turns
- Every request rebuilds the full transcript and sends it as a single prompt

### R7: Minimal flags
- `agent` is assumed installed and available on PATH
- Always call with `--mode=ask`
- `--model` is passed from the composite model name or settings

---

## Design

Implement a `composer-cli` strategy that invokes the Cursor `agent` CLI.

### Command shape
```
agent -p "<serialized transcript>" --model "<model>" --mode=ask
```

### Input serialization
The full transcript is serialized to a single plain-text prompt, then passed via `-p`.

Example (multi-turn):

```
<<<ROLE:SYSTEM>>>
You are a helpful assistant.

<<<ROLE:USER>>>
Hi! What's your name?

<<<ROLE:ASSISTANT>>>
I'm Composer, a language model trained by Cursor. I help with coding tasks in your editor.

<<<ROLE:USER>>>
What can you do?
```

### Output handling
- Treat stdout as the final assistant message (no JSON parsing)
- Strip trailing whitespace only

### Error handling
- Non-zero exit codes and stderr are surfaced in trace logs
- Retry/backoff behavior is left for runtime tuning during CLI integration

### Tests (doc-level guidance)
No automated integration tests are required initially. Provide a manual multi-turn dialog test case using the serialization above and verify:

1. The serialized prompt includes all prior turns.
2. The CLI response is returned as the final assistant message.

---
