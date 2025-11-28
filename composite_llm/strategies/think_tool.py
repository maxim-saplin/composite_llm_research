"""
Think Tool Strategy - Anthropic's "think" tool pattern

This implements the "think" tool as described in:
https://www.anthropic.com/engineering/claude-think-tool

The "think" tool is a no-op tool that provides Claude a dedicated space for 
structured thinking during complex agentic tasks. It's different from 
chain-of-thought prompting or extended thinking:

- Extended thinking: What Claude does BEFORE starting a response
- Think tool: What Claude does DURING a response, between tool calls

Best suited for:
1. Tool output analysis - processing results of previous tool calls
2. Policy-heavy environments - following detailed guidelines
3. Sequential decision making - where each action builds on previous ones

NOT suited for:
- Non-sequential tool calls
- Simple instruction following
- Tasks where default behavior is good enough
"""

import json
import re
from typing import List, Dict, Any, Optional
from .base import BaseStrategy


def strip_thinking_tags(content: Optional[str]) -> str:
    """
    Remove <thinking>...</thinking> or <think>...</think> sections from content.
    
    Some models emit thinking in XML-style tags even when not requested.
    This ensures we don't leak internal reasoning to the final response.
    """
    if not content:
        return ""
    
    # Remove <thinking>...</thinking> blocks (including multiline)
    content = re.sub(r'<thinking>.*?</thinking>\s*', '', content, flags=re.DOTALL)
    # Remove <think>...</think> blocks (including multiline)
    content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
    
    return content.strip()


# The think tool definition following Anthropic's specification
THINK_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "think",
        "description": (
            "Use the tool to think about something. It will not obtain new information "
            "or change the database, but just append the thought to the log. "
            "Use it when complex reasoning or some cache memory is needed."
        ),
        "parameters": {
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
}

# Alternative definition for SWE-bench style tasks
THINK_TOOL_DEFINITION_SWEBENCH = {
    "type": "function", 
    "function": {
        "name": "think",
        "description": (
            "Use the tool to think about something. It will not obtain new information "
            "or make any changes to the repository, but just log the thought. "
            "Use it when complex reasoning or brainstorming is needed. "
            "For example, if you explore the repo and discover the source of a bug, "
            "call this tool to brainstorm several unique ways of fixing the bug, "
            "and assess which change(s) are likely to be simplest and most effective. "
            "Alternatively, if you receive some test results, call this tool to "
            "brainstorm ways to fix the failing tests."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your thoughts."
                }
            },
            "required": ["thought"]
        }
    }
}


def get_think_tool_system_prompt(domain: Optional[str] = None) -> str:
    """
    Returns system prompt instructions for using the think tool effectively.
    
    Based on Anthropic's τ-bench findings, domain-specific prompting significantly
    improves think tool effectiveness.
    
    Args:
        domain: Optional domain hint ('airline', 'retail', 'coding', etc.)
    """
    base_prompt = """## Using the think tool

Before taking any action or responding to the user after receiving tool results, use the think tool as a scratchpad to:
- List the specific rules that apply to the current request
- Check if all required information is collected
- Verify that the planned action complies with all policies
- Iterate over tool results for correctness"""

    if domain == "airline":
        return base_prompt + """

Here are some examples of what to iterate over inside the think tool:
<think_tool_example_1>
User wants to cancel flight ABC123
- Need to verify: user ID, reservation ID, reason
- Check cancellation rules:
  * Is it within 24h of booking?
  * If not, check ticket class and insurance
- Verify no segments flown or are in the past
- Plan: collect missing info, verify rules, get confirmation
</think_tool_example_1>

<think_tool_example_2>
User wants to book 3 tickets to NYC with 2 checked bags each
- Need user ID to check:
  * Membership tier for baggage allowance
  * Which payments methods exist in profile
- Baggage calculation:
  * Economy class × 3 passengers
  * If regular member: 1 free bag each → 3 extra bags = $150
  * If silver member: 2 free bags each → 0 extra bags = $0
  * If gold member: 3 free bags each → 0 extra bags = $0
- Payment rules to verify:
  * Max 1 travel certificate, 1 credit card, 3 gift cards
  * All payment methods must be in profile
  * Travel certificate remainder goes to waste
- Plan:
1. Get user ID
2. Verify membership level for bag fees
3. Check which payment methods in profile and if their combination is allowed
4. Calculate total: ticket price + any bag fees
5. Get explicit confirmation for booking
</think_tool_example_2>"""
    
    elif domain == "coding":
        return base_prompt + """

Here are some examples of what to iterate over inside the think tool:
<think_tool_example_1>
Found a bug in the authentication flow
- Root cause: Token validation happens after authorization check
- Possible fixes:
  1. Reorder validation to happen first (simplest)
  2. Add early return if token invalid
  3. Refactor into middleware (more complex but cleaner)
- Assessment: Option 1 is safest, minimal code change
- Plan: Move token validation before line 45, add test case
</think_tool_example_1>

<think_tool_example_2>
Test failures after refactoring
- 3 tests failing in test_user_service.py
- Analyzing each:
  * test_create_user: expects old return format, need to update assertion
  * test_delete_user: mock not updated for new dependency
  * test_list_users: pagination changed, test data needs adjustment
- Plan: Fix test_create_user first as it's simplest, then others
</think_tool_example_2>"""
    
    return base_prompt


class ThinkToolStrategy(BaseStrategy):
    """
    Implements Anthropic's "think" tool pattern for agentic workflows.
    
    This strategy injects the think tool into the tools list and handles
    think tool calls by simply acknowledging them (no-op). The real value
    comes from Claude using the tool to structure its reasoning.
    
    Usage:
        composite/think_tool/gpt-4o
        
    Optional params:
        - tools: List of other tools to include alongside think
        - domain: Domain hint for optimized prompting ('airline', 'retail', 'coding')
        - use_swebench_definition: Use SWE-bench style tool description
        - include_think_prompt: Add think tool usage instructions to system prompt
    """
    
    def execute(
        self,
        messages: List[Dict[str, str]],
        model_config: str,
        optional_params: Dict[str, Any],
        litellm_params: Dict[str, Any],
    ) -> Any:
        target_model = model_config or "gpt-4o"
        
        # Get configuration options
        user_tools = optional_params.get("tools", [])
        domain = optional_params.get("domain")
        use_swebench = optional_params.get("use_swebench_definition", False)
        include_think_prompt = optional_params.get("include_think_prompt", True)
        max_iterations = optional_params.get("max_iterations", 10)
        
        # Select think tool definition
        think_tool = THINK_TOOL_DEFINITION_SWEBENCH if use_swebench else THINK_TOOL_DEFINITION
        
        # Combine think tool with user-provided tools
        all_tools = [think_tool] + list(user_tools)
        
        # Prepare messages
        working_messages = [m.copy() for m in messages]
        
        # Optionally inject think tool instructions into system prompt
        if include_think_prompt:
            think_prompt = get_think_tool_system_prompt(domain)
            if working_messages and working_messages[0]["role"] == "system":
                working_messages[0]["content"] += f"\n\n{think_prompt}"
            else:
                working_messages.insert(0, {"role": "system", "content": think_prompt})
        
        # Collect all thoughts from think tool calls
        collected_thoughts = []
        
        # Agentic loop - handle tool calls including think
        for _ in range(max_iterations):
            response = self.simple_completion(
                model=target_model,
                messages=working_messages,
                tools=all_tools if all_tools else None,
                **{k: v for k, v in litellm_params.items() if k not in ["tools"]}
            )
            
            message = response.choices[0].message
            
            # Check if there are tool calls
            if not hasattr(message, "tool_calls") or not message.tool_calls:
                # No tool calls, we're done - strip any <thinking> tags from output
                if message.content:
                    message.content = strip_thinking_tags(message.content)
                # Store collected thoughts in reasoning_content field
                if collected_thoughts:
                    message.reasoning_content = "\n\n".join(collected_thoughts)
                return response
            
            # Add assistant message with tool calls to history
            working_messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            
            # Process each tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                
                if tool_name == "think":
                    # Think tool is a no-op - just acknowledge it
                    # The value is in Claude structuring its reasoning
                    # Collect the thought for reasoning_content
                    try:
                        args = json.loads(tool_call.function.arguments)
                        thought = args.get("thought", "")
                        if thought:
                            collected_thoughts.append(thought)
                    except json.JSONDecodeError:
                        pass
                    tool_result = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Thought recorded."
                    }
                else:
                    # For other tools, we'd need a tool executor
                    # For now, return an error indicating tool not implemented
                    tool_result = {
                        "role": "tool", 
                        "tool_call_id": tool_call.id,
                        "content": f"Error: Tool '{tool_name}' execution not implemented in this strategy. "
                                   f"Provide a tool executor via optional_params['tool_executor']."
                    }
                    
                    # Check if user provided a tool executor
                    tool_executor = optional_params.get("tool_executor")
                    if tool_executor and callable(tool_executor):
                        try:
                            result = tool_executor(tool_name, tool_call.function.arguments)
                            tool_result["content"] = str(result)
                        except Exception as e:
                            tool_result["content"] = f"Error executing {tool_name}: {str(e)}"
                
                working_messages.append(tool_result)
        
        # Max iterations reached, strip any <thinking> tags and return last response
        if response and response.choices:
            message = response.choices[0].message
            if message.content:
                message.content = strip_thinking_tags(message.content)
            # Store collected thoughts in reasoning_content field
            if collected_thoughts:
                message.reasoning_content = "\n\n".join(collected_thoughts)
        return response


# Convenience function to get just the tool definition for manual integration
def get_think_tool(swebench_style: bool = False) -> Dict[str, Any]:
    """
    Returns the think tool definition for manual integration into your tools list.
    
    Args:
        swebench_style: If True, use the SWE-bench style description
        
    Returns:
        Tool definition dict compatible with OpenAI/Anthropic tool format
    """
    return THINK_TOOL_DEFINITION_SWEBENCH if swebench_style else THINK_TOOL_DEFINITION

