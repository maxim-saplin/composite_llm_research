from __future__ import annotations

import os

from dotenv import load_dotenv
import litellm

from composite_llm.observability import log_failure, log_success


def bootstrap() -> None:
    load_dotenv()

    # Silence LiteLLM provider list warnings
    litellm.suppress_debug_info = True

    # Ensure Cerebras has a usable API key if only LITELLM_API_KEY is set
    if not os.environ.get("CEREBRAS_API_KEY") and os.environ.get("LITELLM_API_KEY"):
        os.environ["CEREBRAS_API_KEY"] = os.environ["LITELLM_API_KEY"]

    # Observability callbacks
    litellm.success_callback = [log_success]
    litellm.failure_callback = [log_failure]