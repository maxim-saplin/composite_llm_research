import os
import shutil
import textwrap
from typing import Any, cast
from dotenv import load_dotenv
import litellm

from composite_llm.litellm_provider import register_composite_provider
from composite_llm.moa_config import (
    CEREBRAS_MODELS,
    get_council_params,
    get_moa_model,
    get_moa_optional_params,
)
from composite_llm.observability import is_logging_enabled, log_success, log_failure


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


load_dotenv()

# Silence LiteLLM provider list warnings
litellm.suppress_debug_info = True

# Ensure Cerebras has a usable API key if only LITELLM_API_KEY is set
if not os.environ.get("CEREBRAS_API_KEY") and os.environ.get("LITELLM_API_KEY"):
    os.environ["CEREBRAS_API_KEY"] = os.environ["LITELLM_API_KEY"]

# 1. Register Observability Callbacks
litellm.success_callback = [log_success]
litellm.failure_callback = [log_failure]

# 2. Register Custom Provider
register_composite_provider()


def _has_any_api_key() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LITELLM_API_KEY")
        or os.environ.get("CEREBRAS_API_KEY")
    )


def _extract_message_content(resp: Any) -> str:
    if isinstance(resp, dict):
        try:
            return str(resp["choices"][0]["message"].get("content", "") or "")
        except Exception:
            return ""

    try:
        message = resp.choices[0].message
        return str(getattr(message, "content", "") or "")
    except Exception:
        return ""

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
    MODELS = CEREBRAS_MODELS

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
        # MoA Strategy
        {
            "name": "MoA (Agg: 70b, Prop: [8b, Qwen])",
            "model": get_moa_model(),
            "type": "composite",
            "params": get_moa_optional_params(),
        },
        # Council Strategy (LLM Council-style 3-stage flow)
        {
            "name": "Council (Chair: 70b, Council: [8b, Qwen])",
            "model": f"composite/council/{MODELS['llama-70b']}",
            "type": "composite",
            "params": get_council_params(),
        },
    ]

    if shutil.which("agent"):
        configs.append(
            {
                "name": "Composer CLI (agent)",
                "model": "composite/composer-cli/composer-1",
                "type": "composite",
                "params": {"timeout_seconds": 30},
            }
        )
    else:
        print("[demo] Skipping Composer CLI: 'agent' not found on PATH.")

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
                if "cerebras/" in config["model"]:
                    api_key = os.environ.get("CEREBRAS_API_KEY") or os.environ.get(
                        "LITELLM_API_KEY"
                    )
                    if api_key:
                        kwargs["api_key"] = api_key

                resp = cast(
                    Any,
                    litellm.completion(
                        model=config["model"],
                        messages=[{"role": "user", "content": task["prompt"]}],
                        **kwargs,
                    ),
                )

                content = _extract_message_content(resp)

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
    if is_logging_enabled():
        print(
            f"{Colors.CYAN}💡 Tip: Check 'llm_logs.jsonl' for detailed logs{Colors.RESET}"
        )
    else:
        print(
            f"{Colors.CYAN}💡 Tip: Enable logging via config to write llm_logs.jsonl{Colors.RESET}"
        )
    print(
        f"{Colors.CYAN}💡 Run 'streamlit run dashboard.py' for interactive dashboard{Colors.RESET}"
    )


if __name__ == "__main__":
    if not _has_any_api_key():
        print(
            "No OPENAI_API_KEY/LITELLM_API_KEY/CEREBRAS_API_KEY found. "
            "Runs will likely error."
        )

    run_demo()
