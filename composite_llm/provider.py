"""Legacy provider wrapper.

The LiteLLM custom provider implementation now lives in
`composite_llm/litellm_provider.py`. This module is kept only to preserve
backwards compatibility for any external imports.
"""

from .litellm_provider import CompositeLiteLLMProvider


CompositeLLMProvider = CompositeLiteLLMProvider
