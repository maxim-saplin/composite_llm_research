from typing import Any, cast

import litellm

from composite_llm.litellm_provider import register_composite_provider


def test_tool_calls_execute_in_composite_flow() -> None:
    register_composite_provider()

    def tool_executor(name: str, arguments: str) -> str:
        return f"tool:{name}:{arguments}"

    response = cast(Any, litellm.completion(
        model="composite/moa/openai/tool-aggregator",
        messages=[{"role": "user", "content": "Need a tool"}],
        optional_params={
            "proposers": ["openai/proposer-1"],
            "tool_executor": tool_executor,
            "max_tool_iterations": 2,
        },
    ))

    message = response.choices[0].message
    content = str(getattr(message, "content", "") or "")
    assert "Tool-based answer" in content
    reasoning = str(getattr(message, "reasoning_content", "") or "")
    assert "Tool Trace:" in reasoning
