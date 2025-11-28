import litellm
import os
import sys

# Add the current directory to path so we can import composite_llm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from composite_llm.provider import CompositeLLMProvider
from composite_llm.observability import log_success, log_failure

# 1. Register Observability Callbacks
litellm.success_callback = [log_success]
litellm.failure_callback = [log_failure]

# 2. Register Custom Provider
# We can't easily register a "custom_provider" string in litellm global map without monkey patching 
# or using the specific `custom_llm_provider` argument in each call.
# However, we can create a wrapper function or client.
# A cleaner way is to use `litellm.custom_provider_map` if it existed, but litellm 
# supports custom providers via the `custom_llm_provider` param.

composite_provider = CompositeLLMProvider()

def composite_completion(model, messages, **kwargs):
    """
    Wrapper to route calls to our Composite Provider if the model name starts with 'composite/'
    """
    if model.startswith("composite/"):
        # We manually call our provider
        # In a real integration, we might patch litellm or use a specific entry point
        return composite_provider.completion(
            model=model,
            messages=messages,
            model_response=None, # Not used in start
            **kwargs
        )
    else:
        return litellm.completion(model=model, messages=messages, **kwargs)

# Mocking for Demonstration Purposes (if no API key)
if not os.environ.get("OPENAI_API_KEY"):
    print("No OPENAI_API_KEY found. Enabling Mock Mode for demonstration.")
    # We can mock litellm.completion to return dummy data
    def mock_completion(model, messages, **kwargs):
        class MockMessage:
            content = f"Mock response from {model}"
        class MockChoice:
            message = MockMessage()
        class MockResponse:
            choices = [MockChoice()]
            usage = type('obj', (object,), {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15})
        
        return MockResponse()
    
    # Patch litellm.completion used inside strategies
    litellm.completion = mock_completion

# --- DEMO ---

def run_demo():
    print("--- Running Composite LLM Demo ---\n")

    # Scenario 1: Think Strategy
    print("1. Testing 'composite/think/gpt-4o'...")
    try:
        resp = composite_completion(
            model="composite/think/gpt-4o",
            messages=[{"role": "user", "content": "How many r's in strawberry?"}],
            optional_params={"include_thoughts": True} # Custom param we supported
        )
        print(f"Result:\n{resp.choices[0].message.content}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Scenario 2: MoA Strategy
    print("2. Testing 'composite/moa/gpt-4o' (Mixture of Agents)...")
    try:
        resp = composite_completion(
            model="composite/moa/gpt-4o",
            messages=[{"role": "user", "content": "Explain Quantum Entanglement simply."}],
            # We can pass specific proposers
            optional_params={"proposers": ["gpt-3.5-turbo", "claude-3-haiku"]} 
        )
        print(f"Result:\n{resp.choices[0].message.content}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    print("\nCheck 'llm_logs.jsonl' for logs and run 'streamlit run dashboard.py' to view them.")

if __name__ == "__main__":
    run_demo()

