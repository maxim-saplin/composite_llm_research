from typing import Dict, Any, List
import litellm
from .strategies.base import BaseStrategy
# We will import strategies dynamically or register them here
# from .strategies.think import ThinkStrategy
# from .strategies.moa import MoAStrategy


class CompositeLLMProvider:
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        # Lazy loading or registration could happen here
        # For now, let's hardcode imports inside completion or __init__ to avoid circular deps if any

    def _get_strategy(self, strategy_name: str) -> BaseStrategy:
        if strategy_name in self.strategies:
            return self.strategies[strategy_name]

        # Simple dynamic loading based on name
        new_strategy: BaseStrategy
        if strategy_name == "think":
            from .strategies.think import ThinkStrategy

            new_strategy = ThinkStrategy()
        elif strategy_name == "moa":
            from .strategies.moa import MoAStrategy

            new_strategy = MoAStrategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        self.strategies[strategy_name] = new_strategy
        return new_strategy

    def completion(
        self, model: str, messages: List[Dict[str, str]], model_response: Any, **kwargs
    ):
        """
        Handles the completion request for a composite model.
        Expected model format: 'composite/<strategy_name>/<target_model>'
        e.g., 'composite/think/gpt-4o'
        """

        if not model.startswith("composite/"):
            # Fallback to standard litellm if somehow routed here incorrectly
            return litellm.completion(model=model, messages=messages, **kwargs)

        parts = model.split("/")
        if len(parts) < 3:
            raise ValueError(
                f"Invalid composite model string: {model}. Expected 'composite/<strategy>/<model>'"
            )

        strategy_name = parts[1]
        # target_model could be the rest of the string if it contains slashes, strictly it's index 2
        # But let's handle cases like 'composite/think/cerebras/zai-glm-4.6' -> strategy=think, model=cerebras/zai-glm-4.6
        target_model = "/".join(parts[2:])

        strategy = self._get_strategy(strategy_name)

        # Extract optional_params if provided (custom arg used in demo.py)
        optional_params = kwargs.pop("optional_params", {})

        # execute strategy
        return strategy.execute(
            messages=messages,
            model_config=target_model,
            optional_params=optional_params,
            litellm_params=kwargs,
        )
