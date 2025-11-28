Initial goals: (1) understand how to register a custom / composite backend inside LiteLLM so it looks like a normal model; (2) sketch architectures for MoA / think-tool / RLM-style orchestration while preserving standard chat-completions surface (including tools); (3) outline observability + a minimal dashboard. I'll pull in current LiteLLM docs and RLM details, then propose a concrete design.

[3 tools called]

## High-level shape of `composite_llm`

- **Goal**: Make `composite_llm` look like a single normal chat-completions model to LiteLLM, while internally running multi-step pipelines (MoA, think-tool, RLM-style reflection, etc.), plus first-class metrics.
- **Design**: Implement `composite_llm` as a **custom LiteLLM provider** with its own model names (e.g. `composite/moa-v1`), whose `completion` handler orchestrates one or more downstream LiteLLM calls and returns a standard OpenAI-style response.

Below I’ll break it into (1) LiteLLM integration, (2) orchestration modes, (3) tool-call handling, and (4) introspection/dashboard.

---

## 1. How to plug `composite_llm` into LiteLLM

### 1.1 Target UX

From the client’s perspective you want something like:

```python
from litellm import completion

resp = completion(
    model="composite/moa-v1",
    messages=[{"role": "user", "content": "Explain Ruy Lopez plans for Black."}],
)
print(resp.choices[0].message.content)
```

And in `.env` / config:

```bash
LITELLM_MODEL_LIST='[
  {
    "model_name": "composite-moa",
    "litellm_params": { "model": "composite/moa-v1" }
  }
]'
```

So `composite/moa-v1` is just “another” LiteLLM model, but backed by your orchestrator.

### 1.2 Provider registration in LiteLLM

LiteLLM exposes a provider registration mechanism so you can plug in custom backends as if they were built-in providers (see the provider registration docs: [`https://docs.litellm.ai/docs/provider_registration/`](https://docs.litellm.ai/docs/provider_registration/)).

Conceptually, you do the following:

- **Implement a provider module**, e.g. `composite_provider.py`, that exposes the same entry points LiteLLM expects for a provider:
  - `completion(model, messages, **kwargs)`
  - `acompletion(...)` (async)
  - Optionally `streaming` variants if you decide to support streaming.
- **Register your provider** with LiteLLM under a provider name, say `"composite"`, so models like `"composite/moa-v1"` or `"composite/rlm-v1"` dispatch into your code.
- In config / env, map model names to your provider, exactly as you’d do for OpenAI/Anthropic/etc. (LiteLLM already supports mapping many providers behind a unified interface: [`https://github.com/BerriAI/litellm`](https://github.com/BerriAI/litellm)).

Internally, your provider implementation will *itself* call LiteLLM again for the real base models:

```python
from litellm import completion as base_completion

def completion(model, messages, **kwargs):
    # 1. Look up composite mode config for this model (moa, think, rlm, etc.)
    # 2. Run the configured pipeline, using base_completion(...) for actual model calls
    # 3. Return a single OpenAI-style ChatCompletion dict
```

This gives you:

- **Max re-use** of LiteLLM’s provider support (Vertex, Azure, Anthropic, etc.).
- A natural place to attach your own logging/introspection between *outer* `composite_llm` and *inner* base calls.

If you prefer isolation, an alternative is to expose `composite_llm` as its own HTTP service and use LiteLLM’s custom HTTP provider support to talk to it, but the direct provider-module approach is simpler for a first version.

---

## 2. Orchestration modes and building blocks

Think of `composite_llm` as a **small execution engine** over a few reusable primitives, rather than hand-coding each mode.

### 2.1 Core primitives

Define a small internal API that any strategy can use:

- **BaseLLMCall**
  - Inputs: `model_name`, `messages`, `tools`, `tool_choice`, `temperature`, `max_tokens`, etc.
  - Effect: `await base_completion(...)` via LiteLLM.
- **FanOut**
  - Inputs: a list of `BaseLLMCall` specs on the same user query.
  - Effect: run in parallel and collect all responses.
- **Synthesize**
  - Inputs: list of responses from previous calls, plus the original conversation.
  - Effect: invoke a chosen model / prompt to “merge” or “vote” them into one final answer.
- **Reflect**
  - Given a draft answer, call a (possibly different) model to critique it and propose edits.
- **ThinkStep (Scratchpad)**
  - Given tool results or intermediate outputs, run a “think” prompt that writes *internal* reasoning, not exposed to the user (as in Anthropic’s think tool: [`https://www.anthropic.com/engineering/claude-think-tool`](https://www.anthropic.com/engineering/claude-think-tool)).
- **Control primitives**
  - `LoopUntil(condition, max_rounds)`
  - `If(predicate)`

Then each “mode” is just a *configuration* over these primitives.

A simple config could be a JSON/YAML describing the graph:

```yaml
modes:
  moa_v1:
    steps:
      - type: fanout
        experts:
          - model: "gpt-4.1-mini"
            role_prompt: "You are a tactical chess expert..."
          - model: "claude-3.7-sonnet"
            role_prompt: "You focus on long-term strategy..."
      - type: synthesize
        model: "gpt-4.1"
        prompt_template: "You will be provided with several analyses..."
  think_rlm_v1:
    steps:
      - type: base_call
        model: "gpt-4.1-mini"
      - type: think
        model: "gpt-4.1-mini"
      - type: reflect
        model: "gpt-4.1"
        max_rounds: 2
```

Your provider just loads this config and executes it.

### 2.2 Mixture-of-Agents (MoA) mode

MoA is a natural fit:

- **Step 1 – Expert fan-out**:
  - Run N experts (possibly different models, prompts, temperatures) on the same user conversation.
  - Each expert gets:
    - The conversation history.
    - A brief role description (“You are a cautious positional player…”, etc.).
- **Step 2 – Aggregator/synthesizer**:
  - Run a higher-quality model once to synthesize, similar to your `NoN_Synthesizer` idea.
  - Prompt pattern: “You are given multiple candidate answers, some partially incorrect. Produce one accurate, coherent answer in British English…”.
- **Optional Step 3 – Lightweight voting or scoring**:
  - Instead of full synthesis, you can:
    - Ask a critic model to score each expert’s answer.
    - Pick the best one, maybe with minor polishing.
- **Parameters** exposed in config:
  - `num_experts`, expert `models`, `roles`, `temperature`, timeouts, and per-call token budgets.
  - Optionally `max_rounds` for iterative MoA (experts see previous round’s answers).

Because this all happens inside one provider call, clients still see a single `ChatCompletion` with one (or few) final messages.

### 2.3 Think-tool style mode

Inspired by the Anthropic think tool design ([blog](https://www.anthropic.com/engineering/claude-think-tool)):

- **When to add a ThinkStep**:
  - After *tool results* arrive (e.g. long database responses).
  - After a complicated intermediate step (e.g. big fan-out).
- **How it looks internally**:
  - Build an internal “think prompt” that:
    - Lists relevant rules/constraints for the domain.
    - Asks the model to check whether all needed information is present.
    - Asks it to outline a plan (“1. Verify X, 2. Compare Y and Z, 3. Produce final answer”).
  - Store this output in a `scratchpad` key in your trace data.
  - Append a condensed version back into the system messages for the *next* call:
    - e.g. `{"role": "system", "content": "Internal analysis: <summary> Use this only as private reasoning; do not mention 'think step' to user."}`

The user never sees this directly, but you can expose it through your introspection API (useful for debugging and research).

### 2.4 RLM-style reflective mode

Based on the RLM idea from Alex Zhang’s blog ([RLM blog](https://alexzhang13.github.io/blog/2025/rlm/)), you can build a reflective pipeline:

- **Phase 1 – Draft**:
  - Cheap model (or lower temperature): quick first-pass answer.
- **Phase 2 – Reflection**:
  - Same or different model gets:
    - The original question.
    - The draft answer.
    - Instructions: “Identify mistakes, missing cases, vagueness; DO NOT rewrite yet.”
  - Output: a structured list of issues / improvements.
- **Phase 3 – Rewrite**:
  - Stronger model (or same) takes question + draft + reflection and produces a revised final answer.
- **Optional multi-round**:
  - Loop reflection + rewrite a fixed small number of times, or until a “confidence” heuristic passes (length/coverage/etc.).

This maps naturally to primitives: `BaseCall → Reflect → BaseCall`. And it composes nicely with MoA:

- MoA experts → Synthesized draft → RLM-style reflect + rewrite.

---

## 3. Handling tool calls cleanly

You essentially have **three layers**: client ↔ `composite_llm` ↔ base models.

### 3.1 Separation of external vs internal tools

- **External client tools**:
  - The tools that the *client app* defines (DB queries, search, etc.).
  - Only the *outermost* model (the one whose output you return) is allowed to emit these `tool_calls`.
- **Internal tools**:
  - Anything you need just for orchestration (e.g. summarization of logs, small helper models).
  - Implemented entirely inside `composite_llm`; they never appear in the upstream API.

### 3.2 Recommended handling strategy

- **Simple and robust path (recommended for v1)**:
  - For *experts* and intermediate steps:
    - Disable external tools (`tools=[]` or `tool_choice="none"`).
    - Treat the full conversation (including any tool results already executed by the client) as plain text context.
  - For the **final step only** (aggregator / final reflective writer):
    - Allow tools exactly as the client would expect.
    - Let this model decide whether to emit `tool_calls`.
  - Your provider then:
    - If final result is normal content: return it.
    - If final result is a `tool_call`:
      - Return that as-is; the client calls the tool and loops again (from LiteLLM’s POV it’s just a normal tool-using model).

This keeps the surface area identical to standard LiteLLM tool usage, while the internal pipeline looks like pure text reasoning.

### 3.3 More advanced path (later)

If you want underlying experts to call tools too:

- Have `composite_llm` **execute those tool calls on their behalf internally**:
  - Experts produce `tool_calls`.
  - Your code runs the tool, feeds back the result as `tool` messages, continues the expert’s loop.
- You would *not* surface these to the client.
- This is powerful but doubles complexity; I’d keep it as a phase-2 feature.

---

## 4. Introspection, accounting, and dashboard

### 4.1 Data model for traces

For each top-level `composite_llm` request, create a **trace** with:

- **request metadata**:
  - `request_id`, `timestamp`, `user_id` (if any), input model name (`composite/moa-v1`).
- **per-step records** (one per internal call or think step):
  - `step_id`, `parent_step_id`, `type` (`base_call`, `fanout`, `synthesize`, `reflect`, `think`…).
  - For base calls:
    - `provider_model` (e.g. `gpt-4.1-mini`, `claude-3.7-sonnet`).
    - `tokens_prompt`, `tokens_completion`, `latency_ms`.
    - `cost_estimate` if you use LiteLLM’s pricing tables.
  - For think/reflect steps:
    - `scratchpad_summary` (truncated).
- **aggregations**:
  - `total_internal_calls`, `total_tokens_prompt`, `total_tokens_completion`, `total_latency_ms`.
  - Strategy label (e.g. `mode="moa_v1"`).

This can live in:

- A simple **SQLite** DB (great for local dev).
- Or Postgres if you expect more traffic.
- You can also reuse LiteLLM’s logging hooks and pipe into an observability tool like Langfuse if desired ([LiteLLM integrations](https://docs.litellm.ai/docs/integrations/)).

### 4.2 Instrumentation layer

Wrap all base LiteLLM calls with a single helper:

```python
async def traced_completion(step_type, model, messages, **kwargs):
    start = time.perf_counter()
    resp = await base_completion(model=model, messages=messages, **kwargs)
    end = time.perf_counter()

    tokens_in = resp.usage.prompt_tokens
    tokens_out = resp.usage.completion_tokens
    latency_ms = (end - start) * 1000

    log_step(
       step_type=step_type,
       model=model,
       tokens_in=tokens_in,
       tokens_out=tokens_out,
       latency_ms=latency_ms,
       raw_response=maybe_truncated(resp),
    )

    return resp
```

Every orchestration mode uses `traced_completion` instead of calling LiteLLM directly.

Expose a small **HTTP API** from `composite_llm` for introspection:

- `GET /metrics/summary?last_hours=1`
- `GET /traces/{request_id}`
- Maybe `GET /modes` to inspect configured pipelines.

This can be mounted:

- Either as part of the same process (FastAPI app).
- Or as a sidecar that reads from the same DB.

### 4.3 Minimal Streamlit dashboard

Backed by the trace DB / API, a small Streamlit app can show:

- **Overview page**:
  - Total requests over time by `mode`.
  - Avg / p95 latency by `mode`.
  - Avg token usage and cost per request, per `mode` and per underlying base model.
- **Traces page**:
  - Table of recent requests (id, time, mode, user snippet, status, total cost).
  - Clicking a row shows a **timeline of steps**:
    - Fan-out calls in parallel.
    - Think/reflect steps.
    - Final answer.
- **Diagnostics page**:
  - Distribution of “depth” (number of internal calls per request).
  - Which models contribute most to cost.
  - Fraction of requests that used tools (externally visible).

Because the API is yours, you can easily add details like displaying internal think scratchpads for debugging your modes (very valuable for tuning MoA/think/RLM behavior).

---

## 5. Concrete next steps

- **Step 1**: Implement a bare-bones LiteLLM provider called `"composite"` with a trivial mode that just forwards to a single base model; wire it into LiteLLM via provider registration ([docs](https://docs.litellm.ai/docs/provider_registration/)).
- **Step 2**: Add the primitive execution engine and one simple MoA mode (2–3 experts + 1 synthesizer), plus basic per-call logging (tokens + latency).
- **Step 3**: Add a think-style scratchpad step after tool results, following the Anthropic pattern ([think-tool blog](https://www.anthropic.com/engineering/claude-think-tool)).
- **Step 4**: Implement an RLM-style reflective mode inspired by the RLM blog ([RLM blog](https://alexzhang13.github.io/blog/2025/rlm/)), and surface basic trace inspection via a small Streamlit app.

If you’d like, the next iteration can be a concrete provider skeleton (Python) plus a first YAML format for defining modes.