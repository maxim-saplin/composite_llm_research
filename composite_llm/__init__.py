"""Composite LLM package."""

from .litellm_provider import register_composite_provider

__all__ = ["register_composite_provider"]
