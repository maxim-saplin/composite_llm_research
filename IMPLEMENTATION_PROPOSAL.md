# Composite LLM Implementation Proposal

## Scope (Requested)
- Keep strategies: `moa` and `council` only.
- Deliver a proper LiteLLM custom provider integration.
- Add tool-call handling for composite flows.
- Add token accounting for subcalls and aggregate totals.
- Run proposer/reviewer calls in parallel where safe.
- Adjust repo folder organization to reflect supported strategies only.
- Use `reasoning_content` on the final assistant message to expose internal traces.
- Verification: integration tests using a deterministic OpenAI-compatible fake server for both strategies.

## Proposed Implementation

### 1) Provider Integration (LiteLLM custom provider)
**Goal:** direct `litellm.completion(model="composite/<strategy>/<model>")` calls are routed to composite strategies without a demo wrapper.

Planned work:
- Create a provider module (e.g., `composite_llm/litellm_provider.py`) implementing LiteLLM custom provider hooks.
- Register the provider via LiteLLM’s provider registry on import (or via an explicit `register()` function).
- Ensure the provider:
  - Parses `composite/<strategy>/<model>`.
  - Instantiates the strategy and runs it.
  - Returns a `ModelResponse` object consistent with LiteLLM expectations.
- Provide a small shim for backward compatibility in `demo.py` (optional).

### 2) Tool Call Handling
**Goal:** composite strategies should handle tool calls in a predictable, configurable way.

Planned work:
- Introduce a standard tool call handler contract for strategies (e.g., pass `tool_executor` in `optional_params`).
- For MoA/Council:
  - If the *final* model returns tool calls, bubble them to the client (matching the design doc).
  - If `optional_params["tool_executor"]` is provided, optionally execute tool calls and resume (agent loop).
- Add structured trace events for tool calls.
- Document behavior for clients (tool calls are surfaced unless configured to execute internally).

### 3) Token Accounting + Trace Aggregation
**Goal:** provide accurate token/cost reporting for subcalls and expose an internal trace.

Planned work:
- Extend `TraceRecorder` nodes to include `prompt_tokens`, `completion_tokens`, `total_tokens`, and `cost` if available.
- Extract token usage from LiteLLM responses and attach to trace nodes.
- Aggregate totals at the composite call level and attach them to the root trace.
- Embed a summary trace string into `message.reasoning_content` in the final response.
  - Include stage summaries (MoA: proposer list + aggregator; Council: stage1/2/3 summaries).

### 4) Parallel Calls
**Goal:** run subcalls concurrently to reduce latency.

Planned work:
- Use `asyncio.gather` for parallel proposer calls in MoA.
- Use `asyncio.gather` for Stage 1 (council) and Stage 2 (review) calls in Council.
- Provide a fallback to synchronous execution if an event loop is already running.
- Preserve deterministic ordering in traces (e.g., by stable model order, with per-call timing recorded).

### 5) Repo Folder Organization
**Goal:** simplify to the supported strategy set.

Planned work:
- Remove/relocate out-of-scope strategy files.
- Update `composite_llm/strategies/__init__.py` and documentation accordingly.
- Update `DESIGN.md` and `README.md` to reflect new structure and supported strategies.

### 6) Verification (Integration Tests)
**Goal:** deterministic end-to-end tests against a fake OpenAI-compatible server.

Planned work:
- Add a lightweight fake OpenAI-compatible server (e.g., in `tests/fake_openai_server.py`) that:
  - Serves `/v1/chat/completions` with deterministic content based on input.
  - Supports tool call responses deterministically (when prompted).
  - Returns fixed token usage fields.
- Integration tests (pytest):
  - MoA strategy: verifies parallel proposer calls, aggregation prompt composition, trace + reasoning_content, token totals.
  - Council strategy: verifies 3-stage flow, stage summaries, trace + reasoning_content, token totals.
- Use subprocess to start the fake server in test setup, point LiteLLM to the local base URL.

## Task Breakdown (Subagents)

Below is the recommended split so multiple subagents can work in parallel while I coordinate and integrate results.

1) **Provider + Tool Handling (Agent A)**
- Implement LiteLLM custom provider module.
- Add tool call handling contract and behavior.
- Update provider wiring in demo/entry points.

2) **Tracing + Token Accounting (Agent B)**
- Extend `TraceRecorder` and logging to capture usage/cost.
- Implement reasoning_content trace emission for final response.
- Update MoA/Council to record usage per subcall.

3) **Parallelization (Agent C)**
- Introduce async execution paths in MoA and Council.
- Keep stable ordering for deterministic traces.

4) **Repo Organization + Docs (Agent D)**
- Remove out-of-scope strategies and references.
- Update `DESIGN.md`, `README.md`, and folder structure.

5) **Integration Tests + Fake Server (Agent E)**
- Build fake OpenAI-compatible server.
- Write deterministic integration tests for MoA and Council.
- Document test usage and environment.

## Decisions (Confirmed)
- Tool calls bubble to the client by default; internal execution only when `tool_executor` is provided.
- `reasoning_content` will be a compact, human-readable summary with per-stage sections.

## Status (Current)
- Provider: implemented via `composite_llm/litellm_provider.py` and wired in `demo.py`.
- Tool handling: agent loop with `tool_executor` support + tool traces.
- Token accounting: per-subcall usage + aggregate usage in trace.
- Parallel calls: asyncio-based parallelization in MoA + Council.
- Repo org: out-of-scope strategies removed; docs updated.
- Verification: fake OpenAI server + integration tests added.
