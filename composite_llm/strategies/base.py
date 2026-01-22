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
        """Helper to call litellm.completion."""
        import os

        api_key = kwargs.pop("api_key", None)
        if api_key is None:
            api_key = os.environ.get("CEREBRAS_API_KEY") or os.environ.get(
                "LITELLM_API_KEY"
            )

        use_openai = model.startswith("openai/")
        call_kwargs = dict(kwargs)
        if use_openai:
            call_kwargs["api_base"] = os.environ.get("OPENAI_API_BASE")
            call_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY") or api_key

        final_api_key = api_key if not use_openai else call_kwargs.pop("api_key", None)
        return litellm.completion(
            model=model,
            messages=messages,
            api_key=final_api_key,
            **call_kwargs,
        )
