[7 tools called]

Let me gather one more piece of specific technical details and then compile the comprehensive research.

[2 tools called]

Now I have gathered comprehensive information. Let me compile this into a detailed research document for your `composite_llm` idea.

---

# Research Report: `composite_llm` Extension for LiteLLM

## Executive Summary

Your `composite_llm` concept is highly viable and aligns with emerging patterns in LLM orchestration. The key insight is that **behind a standard chat completions interface**, you can orchestrate complex multi-model pipelines, reasoning strategies, and ensembles—all transparent to the client code.

---

## Part 1: LiteLLM Integration Mechanisms

### Option A: Custom Provider Registration (Recommended)

LiteLLM allows registering custom providers through their **Provider Registration** API ([docs.litellm.ai](https://docs.litellm.ai/docs/provider_registration/)):

```python
# Conceptual structure for composite_llm provider
from litellm.llms.base_llm.chat.transformation import BaseLLMException, BaseConfig

class CompositeLLMConfig(BaseConfig):
    """Configuration for composite_llm strategies"""
    
    def validate_environment(self, api_key, headers, model, messages, **kwargs):
        # Validate that underlying models are configured
        pass
    
    def transform_request(self, model, messages, **kwargs):
        # Transform input for the composite strategy
        pass
    
    def transform_response(self, raw_response, **kwargs):
        # Return standard OpenAI-format response
        pass
```

**Key integration points:**
1. **Model string prefix**: Use `composite/moa-gpt4-claude` pattern
2. **Environment variables**: Define in `.env` like `COMPOSITE_LLM_STRATEGY=moa`
3. **Config validation**: Validate all underlying model APIs are accessible

### Option B: OpenAI-Compatible Proxy Server

An alternative is to create a standalone FastAPI server that:
- Exposes `/v1/chat/completions` endpoint (OpenAI-compatible)
- Uses `litellm` internally to call underlying models
- Returns standard responses

```python
# Simplified architecture
from fastapi import FastAPI
import litellm

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    strategy = get_strategy(request.model)  # e.g., "composite/moa"
    
    # Execute composite strategy (MoA, Think, RLM, etc.)
    result = await strategy.execute(request.messages)
    
    # Return standard OpenAI format
    return ChatCompletionResponse(
        choices=[Choice(message=result.message)],
        usage=result.aggregated_usage
    )
```

### Option C: LiteLLM Router Extension

Leverage LiteLLM's built-in **Router** class for load balancing and extend it:

```python
from litellm import Router

# LiteLLM Router already supports multiple models
router = Router(
    model_list=[
        {"model_name": "gpt-4", "litellm_params": {...}},
        {"model_name": "claude-3", "litellm_params": {...}},
    ],
    routing_strategy="simple-shuffle"  # or custom
)
```

You could extend this with a custom `routing_strategy` that implements MoA/ensemble logic.

---

## Part 2: Orchestration Strategies (Building Blocks)

### Strategy 1: Mixture-of-Agents (MoA)

**Reference**: [Together.ai MoA](https://github.com/togethercomputer/MoA) — achieved 65.1% on AlpacaEval 2.0, surpassing GPT-4o's 57.5%.

**Architecture:**

```
Layer 0 (Proposers):     [Model_A]  [Model_B]  [Model_C]
                              ↓          ↓          ↓
                           Response_A  Response_B  Response_C
                              ↘          ↓          ↙
Layer 1 (Aggregator):          [Aggregator Model]
                                      ↓
                              Final Response
```

**Your MoA implementation** (from `custom_agents.py`):

```583:591:custom_agents.py
        self._name = "NoN_Synthesizer"
        self._human_input_mode="NEVER",
        self._system_message = """\
You will be provided with a set of responses from various open-source models to the latest user query.
Your task is to synthesize these responses into a single, high-quality response in British English spelling.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.
Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction.
Ensure your response is well-structured, coherent and adheres to the highest standards of accuracy and reliability.
"""
```

**Implementation pattern:**

```python
class MoAStrategy:
    def __init__(self, proposer_models: list[str], aggregator_model: str, layers: int = 1):
        self.proposer_models = proposer_models
        self.aggregator_model = aggregator_model
        self.layers = layers
    
    async def execute(self, messages: list[dict]) -> CompositeResult:
        responses = []
        
        # Layer 0: Parallel proposer calls
        tasks = [
            litellm.acompletion(model=model, messages=messages)
            for model in self.proposer_models
        ]
        proposer_responses = await asyncio.gather(*tasks)
        
        # Aggregate responses
        aggregation_prompt = self._build_aggregation_prompt(
            original_query=messages[-1]["content"],
            responses=[r.choices[0].message.content for r in proposer_responses]
        )
        
        final = await litellm.acompletion(
            model=self.aggregator_model,
            messages=[{"role": "user", "content": aggregation_prompt}]
        )
        
        return CompositeResult(
            message=final.choices[0].message,
            usage=self._aggregate_usage(proposer_responses + [final])
        )
```

---

### Strategy 2: Think Tool (Anthropic)

**Reference**: [Anthropic Engineering Blog](https://www.anthropic.com/engineering/claude-think-tool)

**Key insight**: Unlike extended thinking (pre-response), the Think Tool is for **mid-chain reasoning**—particularly valuable in long tool call sequences.

**Implementation:**

```python
THINK_TOOL = {
    "name": "think",
    "description": """Use this tool to think about something. It will not obtain 
    new information or change the database, but just append the thought to the log. 
    Use it when complex reasoning or some cache memory is needed.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "A thought to think about."
            }
        },
        "required": ["thought"]
    }
}

class ThinkToolStrategy:
    """Wraps any model with a think tool for structured reasoning"""
    
    def __init__(self, model: str, think_prompt: str = None):
        self.model = model
        self.think_prompt = think_prompt or self._default_think_prompt()
    
    async def execute(self, messages: list[dict], tools: list[dict] = None):
        # Inject think tool
        all_tools = [THINK_TOOL] + (tools or [])
        
        # Add think prompt to system message
        enhanced_messages = self._inject_think_prompt(messages)
        
        # Execute with tool loop
        while True:
            response = await litellm.acompletion(
                model=self.model,
                messages=enhanced_messages,
                tools=all_tools
            )
            
            if response.choices[0].finish_reason == "tool_calls":
                tool_calls = response.choices[0].message.tool_calls
                
                for tc in tool_calls:
                    if tc.function.name == "think":
                        # Log thought but don't return result to user
                        self._log_thought(tc.function.arguments)
                        # Continue conversation with empty tool result
                        enhanced_messages.append({"role": "tool", "content": "Thought recorded.", "tool_call_id": tc.id})
                    else:
                        # Execute real tool
                        result = await self._execute_tool(tc)
                        enhanced_messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})
            else:
                break
        
        return response
```

**When to use:**
- Complex multi-step tool call chains
- Policy-heavy environments requiring compliance checks
- Sequential decisions where each step builds on previous ones

**τ-Bench results**: 54% improvement on airline domain with optimized prompting.

---

### Strategy 3: Recursive Language Models (RLM)

**Reference**: [Alex Zhang's RLM Blog](https://alexzhang13.github.io/blog/2025/rlm/) and [GitHub](https://github.com/alexzhang13/rlm)

**Key insight**: RLM allows models to **recursively call themselves** (or other LLMs) to handle unbounded context and complex reasoning—without stuffing everything into a single context window.

**Architecture:**

```
User Query → Root LM (sees only query, not full context)
                 ↓
            REPL Environment (Python notebook-like)
                 ↓
           [Context stored in environment variables]
                 ↓
           LM can: read_lines(start, end)
                   grep(pattern)
                   call_llm(sub_query, context_slice)  ← Recursive!
                   FINAL(answer)
```

**Results:**
- RLM(GPT-4-mini) **outperforms GPT-4** on OOLONG benchmark by >33% on 128k+ token contexts
- Handles 10M+ tokens effectively by never clogging the root model's context

**Emergent strategies:**
1. **Peeking**: Root LM samples a few entries to understand structure
2. **Grepping**: Keyword/regex search to narrow down relevant lines
3. **Partition + Map**: Chunk context, process each chunk, reduce results
4. **Summarization**: Recursive summarization of context subsets

**Implementation pattern:**

```python
class RLMStrategy:
    def __init__(self, root_model: str, recursive_model: str = None, max_depth: int = 3):
        self.root_model = root_model
        self.recursive_model = recursive_model or root_model
        self.max_depth = max_depth
    
    async def execute(self, query: str, context: str) -> CompositeResult:
        # Store context in environment (not in prompt)
        env = REPLEnvironment(context)
        
        # Root model interacts via tools, never sees full context
        tools = [
            {"name": "read_lines", ...},
            {"name": "grep", ...},
            {"name": "call_llm", ...},  # Recursive call
            {"name": "FINAL", ...}
        ]
        
        messages = [{"role": "user", "content": f"Answer this query: {query}\n\nContext is available via tools."}]
        
        return await self._execute_rlm_loop(messages, env, tools, depth=0)
```

---

### Strategy 4: Hybrid Strategies

You can **compose** strategies:

```python
class HybridStrategy:
    """Combine MoA proposers with Think-enabled aggregator"""
    
    async def execute(self, messages):
        # Step 1: MoA proposers (parallel)
        moa = MoAStrategy(proposers=["gpt-4", "claude-3", "gemini-pro"])
        proposals = await moa.get_proposals(messages)
        
        # Step 2: Think-enabled synthesis
        think = ThinkToolStrategy(model="claude-3-sonnet")
        synthesis_prompt = f"Synthesize these proposals: {proposals}"
        
        return await think.execute([{"role": "user", "content": synthesis_prompt}])
```

---

## Part 3: Handling Tool Calls

**Challenge**: When `composite_llm` receives messages with tool calls, how should it behave?

### Option A: Pass-Through Mode

If client passes tools, the composite strategy should:
1. Inject additional tools (like `think`) if the strategy requires
2. Route tool execution back to client (or internal handlers)
3. Aggregate tool usage across all underlying models

```python
async def execute_with_tools(self, messages, tools, tool_executor):
    # Each underlying model call may generate tool calls
    # Aggregate them, execute, continue conversation
    pass
```

### Option B: Tool Interception

For strategies like **Think Tool**, intercept specific tool calls internally:

```python
if tool_call.function.name == "think":
    # Internal processing, don't expose to client
    self._log_thought(tool_call)
    continue
else:
    # Pass to client's tool executor
    result = await tool_executor(tool_call)
```

### Option C: Tool Aggregation

For MoA, multiple proposers might suggest different tool calls:

```python
# Collect tool calls from all proposers
all_tool_calls = []
for response in proposer_responses:
    if response.choices[0].message.tool_calls:
        all_tool_calls.extend(response.choices[0].message.tool_calls)

# Deduplicate or let aggregator decide
aggregator_decision = await self._decide_tools(all_tool_calls)
```

---

## Part 4: Introspection & Observability

### Data Model for Tracking

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class LLMCall:
    call_id: str
    model: str
    strategy: str
    parent_call_id: Optional[str]  # For nested/recursive calls
    
    # Timing
    start_time: datetime
    end_time: datetime
    latency_ms: float
    
    # Tokens
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    # Cost (if available)
    cost_usd: Optional[float]
    
    # Content (optional, for debugging)
    input_messages: Optional[list] = None
    output_content: Optional[str] = None

@dataclass
class CompositeTrace:
    trace_id: str
    strategy: str
    total_calls: int
    calls: list[LLMCall]
    
    # Aggregated metrics
    total_latency_ms: float
    total_tokens: int
    total_cost_usd: float
    
    # Strategy-specific
    metadata: dict = field(default_factory=dict)  # e.g., {"layers": 2, "proposers": [...]}
```

### LiteLLM Callbacks Integration

LiteLLM provides built-in callback mechanisms:

```python
import litellm
from litellm.integrations.custom_logger import CustomLogger

class CompositeLLMLogger(CustomLogger):
    def __init__(self, trace_store):
        self.trace_store = trace_store
    
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        call = LLMCall(
            call_id=kwargs.get("litellm_call_id"),
            model=kwargs.get("model"),
            strategy=kwargs.get("metadata", {}).get("strategy"),
            start_time=start_time,
            end_time=end_time,
            latency_ms=(end_time - start_time).total_seconds() * 1000,
            prompt_tokens=response_obj.usage.prompt_tokens,
            completion_tokens=response_obj.usage.completion_tokens,
            total_tokens=response_obj.usage.total_tokens,
        )
        self.trace_store.add_call(call)

# Register globally
litellm.callbacks = [CompositeLLMLogger(trace_store)]
```

### Existing Observability Integrations

LiteLLM already integrates with:
- **OpenTelemetry** → Jaeger, Zipkin, Datadog, New Relic
- **MLflow** → Experiment tracking
- **Prometheus** → Metrics scraping
- **Langfuse** → LLM-specific observability
- **Lunary** → Prompt analytics

### Streamlit Dashboard Concept

```python
import streamlit as st
from composite_llm import TraceStore

st.title("composite_llm Dashboard")

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Calls", store.total_calls)
col2.metric("Total Tokens", f"{store.total_tokens:,}")
col3.metric("Avg Latency", f"{store.avg_latency_ms:.0f}ms")
col4.metric("Total Cost", f"${store.total_cost:.2f}")

# Strategy breakdown
st.subheader("Strategy Performance")
df = store.get_strategy_stats()
st.bar_chart(df.set_index("strategy")[["avg_latency_ms", "avg_tokens"]])

# Call trace explorer
st.subheader("Recent Traces")
for trace in store.recent_traces(limit=10):
    with st.expander(f"Trace {trace.trace_id} - {trace.strategy}"):
        st.json(trace.to_dict())
        
        # Visualize call hierarchy (for RLM recursive calls)
        if trace.strategy == "rlm":
            st.graphviz_chart(trace.to_graphviz())
```

---

## Part 5: Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Client Code (unchanged)                       │
│                 litellm.completion(model="composite/moa-v1")     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      composite_llm                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Strategy Router                                          │   │
│  │  - Parse model string (composite/moa-v1, composite/rlm)   │   │
│  │  - Load strategy config                                   │   │
│  │  - Initialize trace context                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐    │
│  │  Strategies (pluggable)                                  │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐    │    │
│  │  │   MoA   │ │  Think  │ │   RLM   │ │   Custom    │    │    │
│  │  │ Strategy│ │ Strategy│ │ Strategy│ │  Strategy   │    │    │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘    │    │
│  └───────┼──────────┼──────────┼──────────────┼────────────┘    │
│          │          │          │              │                  │
│  ┌───────▼──────────▼──────────▼──────────────▼────────────┐    │
│  │  LiteLLM Layer (unified API to underlying models)        │    │
│  │  - GPT-4, Claude, Gemini, Llama, etc.                    │    │
│  │  - Built-in retry, fallback, caching                     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐    │
│  │  Observability Layer                                     │    │
│  │  - Trace aggregation                                     │    │
│  │  - Token/cost accounting                                 │    │
│  │  - Callback hooks (Prometheus, OpenTelemetry, etc.)      │    │
│  └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  Streamlit Dashboard                             │
│  - Real-time metrics                                             │
│  - Trace explorer                                                │
│  - Cost analysis                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key References

| Topic                             | Resource                                                                                                                                               |
|-----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| LiteLLM Provider Registration     | [docs.litellm.ai/docs/provider_registration](https://docs.litellm.ai/docs/provider_registration/)                                                      |
| Together MoA Implementation       | [github.com/togethercomputer/MoA](https://github.com/togethercomputer/MoA)                                                                             |
| Anthropic Think Tool              | [anthropic.com/engineering/claude-think-tool](https://www.anthropic.com/engineering/claude-think-tool)                                                 |
| RLM Blog & Code                   | [alexzhang13.github.io/blog/2025/rlm](https://alexzhang13.github.io/blog/2025/rlm/) / [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm) |
| SMoA Paper                        | [arxiv.org/abs/2411.03284](https://arxiv.org/abs/2411.03284)                                                                                           |
| TUMIX (Tool-Use Mixture)          | [arxiv.org/abs/2510.01279](https://arxiv.org/abs/2510.01279)                                                                                           |
| HALO (Hierarchical Orchestration) | [arxiv.org/abs/2505.13516](https://arxiv.org/abs/2505.13516)                                                                                           |

---

## Next Steps

1. **Start with Option B** (standalone proxy) for rapid prototyping
2. **Implement MoA first** — simplest strategy, proven results
3. **Add Think Tool** as an optional enhancement layer
4. **Build minimal Streamlit dashboard** early for debugging
5. **Later**: Register as proper LiteLLM custom provider for seamless integration

Would you like me to draft a concrete implementation skeleton for any of these components?