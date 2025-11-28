from abc import ABC, abstractmethod
from typing import List, Dict, Any
import litellm


class BaseStrategy(ABC):
    @abstractmethod
    def execute(
        self,
        messages: List[Dict[str, str]],
        model_config: str,
        optional_params: Dict[str, Any],
        litellm_params: Dict[str, Any],
    ) -> Any:
        pass

    def simple_completion(self, model: str, messages: List[Dict[str, str]], **kwargs):
        """Helper to call litellm.completion"""
        import os
        # Ensure we don't infinitely recurse if the user passes a composite model as a sub-model
        # (unless that's intended)

        # Inject API key if available in env and not in kwargs
        if "api_key" not in kwargs:
            api_key = os.environ.get("LITELLM_API_KEY")
            if api_key:
                kwargs["api_key"] = api_key

        return litellm.completion(model=model, messages=messages, **kwargs)
