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
        return litellm.completion(
            model=model,
            messages=messages,
            **kwargs,
        )
