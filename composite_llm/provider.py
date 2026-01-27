"""Legacy provider wrapper.

The LiteLLM custom provider implementation now lives in
`composite_llm/litellm_provider.py`. This module is kept only to preserve
backwards compatibility for any external imports.

For LiteLLM proxy usage, prefer YAML-first loading via
`custom_provider_map` pointing at `composite_llm.litellm_provider:get_composite_provider`.
"""

from .litellm_provider import CompositeLiteLLMProvider


CompositeLLMProvider = CompositeLiteLLMProvider
