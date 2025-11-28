import os
import textwrap
from dotenv import load_dotenv
import litellm

from composite_llm.provider import CompositeLLMProvider
from composite_llm.observability import log_success, log_failure


# ANSI colors for pretty terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def format_response(content: str, width: int = 80) -> str:
    """Format response text with word wrapping and indentation."""
    lines = content.split("\n")
    formatted_lines = []
    for line in lines:
        if line.strip():
            wrapped = textwrap.fill(line, width=width, subsequent_indent="  ")
            formatted_lines.append(wrapped)
        else:
            formatted_lines.append("")
    return "\n".join(formatted_lines)


# Load environment variables
load_dotenv()

# 1. Register Observability Callbacks
litellm.success_callback = [log_success]
litellm.failure_callback = [log_failure]

# 2. Register Custom Provider
composite_provider = CompositeLLMProvider()


def composite_completion(model, messages, **kwargs):
    """
    Wrapper to route calls to our Composite Provider if the model name starts with 'composite/'
    """
    if model.startswith("composite/"):
        return composite_provider.completion(
            model=model,
            messages=messages,
            model_response=None,
            **kwargs,
        )
    else:
        # Pass LITELLM_API_KEY as api_key if not set otherwise
        api_key = kwargs.get("api_key")
        if not api_key:
            api_key = os.environ.get("LITELLM_API_KEY")
            if api_key:
                kwargs["api_key"] = api_key

        return litellm.completion(model=model, messages=messages, **kwargs)


# Mocking for Demonstration Purposes (if no API key)
if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("LITELLM_API_KEY"):
    print(
        "No OPENAI_API_KEY or LITELLM_API_KEY found. Enabling Mock Mode for demonstration."
    )

    def mock_completion(model, messages, **kwargs):
        class MockMessage:
            content = f"Mock response from {model}"

        class MockChoice:
            message = MockMessage()

        class MockResponse:
            choices = [MockChoice()]
            usage = type(
                "obj",
                (object,),
                {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        return MockResponse()

    litellm.completion = mock_completion

# --- DEMO ---


def run_demo():
    print("--- Running Composite LLM Demo ---\n")

    tasks = [
        {
            "id": "count_rs",
            "prompt": "How many r's in strawberry?",
            "category": "Logic/Counting",
        },
        # {
        #     "id": "quantum",
        #     "prompt": "Explain Quantum Entanglement simply.",
        #     "category": "Explanation"
        # }
    ]

    # Available Cerebras models
    MODELS = {
        "llama-70b": "cerebras/llama-3.3-70b",
        "llama-8b": "cerebras/llama3.1-8b",
        "qwen-32b": "cerebras/qwen-3-32b",
    }

    configs = [
        # Baseline Models
        {
            "name": "Llama-3.3-70b (Base)",
            "model": MODELS["llama-70b"],
            "type": "baseline",
        },
        {
            "name": "Llama-3.1-8b (Base)",
            "model": MODELS["llama-8b"],
            "type": "baseline",
        },
        # Chain of Thought Strategy (two-step prompting with explicit reasoning)
        # Note: "think" is kept as alias for backwards compatibility
        {
            "name": "CoT + Llama-70b",
            "model": f"composite/cot/{MODELS['llama-70b']}",
            "type": "composite",
            "params": {},
        },
        {
            "name": "CoT + Llama-8b",
            "model": f"composite/cot/{MODELS['llama-8b']}",
            "type": "composite",
            "params": {},
        },
        # Think Tool Strategy (Anthropic's pattern for agentic workflows)
        # Best for: tool output analysis, policy compliance, sequential decisions
        # See: https://www.anthropic.com/engineering/claude-think-tool
        {
            "name": "ThinkTool + Llama-70b",
            "model": f"composite/think_tool/{MODELS['llama-70b']}",
            "type": "composite",
            "params": {"include_think_prompt": True},
        },
        # MoA Strategy
        {
            "name": "MoA (Agg: 70b, Prop: [8b, Qwen])",
            "model": f"composite/moa/{MODELS['llama-70b']}",
            "type": "composite",
            "params": {"proposers": [MODELS["llama-8b"], MODELS["qwen-32b"]]},
        },
    ]

    results = []

    for task in tasks:
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.HEADER}📝 Task: {task['prompt']}{Colors.RESET}")
        print(f"{Colors.DIM}   Category: {task['category']}{Colors.RESET}")
        print(f"{Colors.HEADER}{'═' * 70}{Colors.RESET}")

        for config in configs:
            config_type = config.get("type", "unknown")
            if config_type == "baseline":
                icon = "🔹"
                color = Colors.BLUE
            elif "CoT" in config["name"]:
                icon = "🧠"
                color = Colors.CYAN
            elif "ThinkTool" in config["name"]:
                icon = "💭"
                color = Colors.YELLOW
            elif "MoA" in config["name"]:
                icon = "🤝"
                color = Colors.GREEN
            else:
                icon = "▶"
                color = Colors.RESET

            print(f"\n{color}{Colors.BOLD}{icon} {config['name']}{Colors.RESET}")
            print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")

            try:
                # Prepare args
                kwargs = {}
                if "params" in config:
                    kwargs["optional_params"] = config["params"]

                resp = composite_completion(
                    model=config["model"],
                    messages=[{"role": "user", "content": task["prompt"]}],
                    **kwargs,
                )

                content = resp.choices[0].message.content

                # Format and display the full response
                formatted = format_response(content, width=70)
                print(formatted)

                # Save for summary
                results.append(
                    {
                        "config": config["name"],
                        "task": task["prompt"],
                        "response": content,
                        "success": True,
                    }
                )

            except Exception as e:
                print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")
                results.append(
                    {
                        "config": config["name"],
                        "task": task["prompt"],
                        "response": f"ERROR: {str(e)}",
                        "success": False,
                    }
                )

    # Summary section
    print(f"\n\n{Colors.BOLD}{Colors.HEADER}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}📊 SUMMARY{Colors.RESET}")
    print(f"{Colors.HEADER}{'═' * 70}{Colors.RESET}\n")

    for task in tasks:
        task_results = [r for r in results if r["task"] == task["prompt"]]
        print(f"{Colors.BOLD}Task: {task['prompt']}{Colors.RESET}\n")

        for r in task_results:
            status = (
                f"{Colors.GREEN}✓{Colors.RESET}"
                if r["success"]
                else f"{Colors.RED}✗{Colors.RESET}"
            )
            # Extract just the final answer (last line or last portion)
            answer = r["response"].strip().split("\n")[-1]
            if len(answer) > 100:
                answer = "..." + answer[-100:]
            print(f"  {status} {Colors.BOLD}{r['config']:30}{Colors.RESET} → {answer}")
        print()

    print(f"{Colors.DIM}{'─' * 70}{Colors.RESET}")
    print(
        f"{Colors.CYAN}💡 Tip: Check 'llm_logs.jsonl' for detailed logs{Colors.RESET}"
    )
    print(
        f"{Colors.CYAN}💡 Run 'streamlit run dashboard.py' for interactive dashboard{Colors.RESET}"
    )


if __name__ == "__main__":
    run_demo()
