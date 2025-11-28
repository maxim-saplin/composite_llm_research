import os
import pandas as pd
from dotenv import load_dotenv
import litellm

from composite_llm.provider import CompositeLLMProvider
from composite_llm.observability import log_success, log_failure

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
        # Think Strategy (System 2)
        {
            "name": "Think + Llama-70b",
            "model": f"composite/think/{MODELS['llama-70b']}",
            "type": "composite",
            "params": {"include_thoughts": True},
        },
        {
            "name": "Think + Llama-8b",
            "model": f"composite/think/{MODELS['llama-8b']}",
            "type": "composite",
            "params": {"include_thoughts": True},
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
        print(f"\n=== Task: {task['prompt']} ({task['category']}) ===")

        for config in configs:
            print(f"\n>> Running {config['name']}...")
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

                # For display in terminal, we might want to trim or show structure
                print(
                    f"[{config['name']}] Response:\n{content[:500]}..."
                    if len(content) > 500
                    else f"[{config['name']}] Response:\n{content}"
                )

                # Save for summary
                # If it's a Think strategy, content includes thoughts. We might want to separate them for the table if possible,
                # but currently they are merged text.
                results.append(
                    {
                        "Task": task["prompt"],
                        "Configuration": config["name"],
                        "Response Snippet": content[:100].replace("\n", " ") + "...",
                    }
                )

            except Exception as e:
                print(f"Error running {config['name']}: {e}")
                results.append(
                    {
                        "Task": task["prompt"],
                        "Configuration": config["name"],
                        "Response Snippet": f"ERROR: {str(e)}",
                    }
                )

    print("\n\n=== Summary Table ===")
    df = pd.DataFrame(results)
    # Use tabulate format for nicer output if pandas supports it or just print
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))

    print(
        "\nCheck 'llm_logs.jsonl' for logs and run 'streamlit run dashboard.py' to view them."
    )


if __name__ == "__main__":
    run_demo()
