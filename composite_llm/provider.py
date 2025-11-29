from typing import Dict, Any, List
from datetime import datetime

import litellm

from .strategies.base import BaseStrategy
from .trace import TraceRecorder, get_user_request_preview
from .observability import log_success


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
            # Legacy alias - maps to chain_of_thought for backwards compatibility
            from .strategies.chain_of_thought import ChainOfThoughtStrategy

            new_strategy = ChainOfThoughtStrategy()
        elif strategy_name == "cot" or strategy_name == "chain_of_thought":
            from .strategies.chain_of_thought import ChainOfThoughtStrategy

            new_strategy = ChainOfThoughtStrategy()
        elif strategy_name == "think_tool":
            # Anthropic's "think" tool pattern for agentic workflows
            from .strategies.think_tool import ThinkToolStrategy

            new_strategy = ThinkToolStrategy()
        elif strategy_name == "moa":
            from .strategies.moa import MoAStrategy

            new_strategy = MoAStrategy()
        elif strategy_name == "council":
            from .strategies.council import CouncilStrategy

            new_strategy = CouncilStrategy()
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
        if optional_params is None:
            optional_params = {}

        # Create a trace recorder for this composite call
        user_preview = get_user_request_preview(messages)
        trace_recorder = TraceRecorder(
            strategy=strategy_name,
            root_model=model,
            user_request_preview=user_preview,
        )

        # Root node: user request
        root_node_id = trace_recorder.add_node(
            step_type="user_request",
            parent_id=None,
            model=model,
            role="user",
            content_preview=user_preview,
        )

        # Make trace information available to strategies
        optional_params = dict(optional_params)
        optional_params["trace_recorder"] = trace_recorder
        optional_params["trace_root_node_id"] = root_node_id

        start_time = datetime.now()

        # execute strategy
        final_response = strategy.execute(
            messages=messages,
            model_config=target_model,
            optional_params=optional_params,
            litellm_params=kwargs,
        )

        # Log a composite-level entry with the full trace graph
        end_time = datetime.now()
        try:
            log_kwargs = {
                "model": model,  # composite model string
                "messages": messages,
                "trace": trace_recorder.to_dict(),
            }
            log_success(log_kwargs, final_response, start_time, end_time)
        except Exception:
            # Logging should never break the main flow
            pass

        return final_response
