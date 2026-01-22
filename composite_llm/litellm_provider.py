from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import litellm

from .observability import log_success
from .strategies.base import BaseStrategy
from .trace import TraceRecorder, get_user_request_preview


def _parse_composite_model(model: str) -> Tuple[str, str, str, str]:
    parts = model.split("/")
    if model.startswith("composite/"):
        if len(parts) < 3:
            raise ValueError(
                f"Invalid composite model string: {model}. Expected 'composite/<strategy>/<model>'."
            )

        strategy_name = parts[1]
        target_model = "/".join(parts[2:])
        root_model = model
        return strategy_name, target_model, root_model, "composite"

    if parts and parts[0] in {"moa", "council"}:
        if len(parts) < 2:
            raise ValueError(
                f"Invalid composite model string: {model}. Expected '<strategy>/<model>'."
            )
        strategy_name = parts[0]
        target_model = "/".join(parts[1:])
        root_model = f"composite/{model}"
        return strategy_name, target_model, root_model, "composite"

    raise ValueError(
        f"Invalid composite model string: {model}. Expected 'composite/<strategy>/<model>'."
    )


def _normalize_tool_calls(tool_calls: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            call_id = tool_call.get("id")
            call_type = tool_call.get("type") or "function"
            function = tool_call.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments")
        else:
            call_id = getattr(tool_call, "id", None)
            call_type = getattr(tool_call, "type", None) or "function"
            function = getattr(tool_call, "function", None)
            name = getattr(function, "name", None) if function else None
            arguments = getattr(function, "arguments", None) if function else None

        if not name:
            continue

        normalized.append(
            {
                "id": call_id or f"call_{len(normalized) + 1}",
                "type": call_type,
                "function": {"name": name, "arguments": arguments or "{}"},
            }
        )

    return normalized


def _extract_tool_calls(message: Any) -> List[Dict[str, Any]]:
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls") or []
    else:
        tool_calls = getattr(message, "tool_calls", None) or []
    return _normalize_tool_calls(tool_calls)


class CompositeLiteLLMProvider:
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}

    def _get_strategy(self, strategy_name: str) -> BaseStrategy:
        if strategy_name in self.strategies:
            return self.strategies[strategy_name]

        new_strategy: BaseStrategy
        if strategy_name == "moa":
            from .strategies.moa import MoAStrategy

            new_strategy = MoAStrategy()
        elif strategy_name == "council":
            from .strategies.council import CouncilStrategy

            new_strategy = CouncilStrategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        self.strategies[strategy_name] = new_strategy
        return new_strategy

    def completion(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Any:
        if not (model.startswith("composite/") or model.startswith("moa/") or model.startswith("council/")):
            return litellm.completion(model=model, messages=messages, **kwargs)

        strategy_name, target_model, root_model, provider_name = _parse_composite_model(
            model
        )
        root_model_name = root_model
        strategy = self._get_strategy(strategy_name)

        raw_optional_params = kwargs.pop("optional_params", {}) or {}
        optional_params = raw_optional_params
        passthrough_optional_params: Dict[str, Any] = {}
        if (
            isinstance(raw_optional_params, dict)
            and "optional_params" in raw_optional_params
            and isinstance(raw_optional_params["optional_params"], dict)
        ):
            optional_params = raw_optional_params["optional_params"]
            passthrough_optional_params = {
                k: v for k, v in raw_optional_params.items() if k != "optional_params"
            }

        tool_executor = optional_params.get("tool_executor")
        max_tool_iterations = optional_params.get("max_tool_iterations", 5)
        if "optional_params" in kwargs:
            kwargs.pop("optional_params")

        allowed_kwargs = {"api_key", "api_base", "timeout", "headers", "max_retries"}
        safe_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}

        user_preview = get_user_request_preview(messages)
        trace_recorder = TraceRecorder(
            strategy=strategy_name,
            root_model=root_model,
            user_request_preview=user_preview,
        )
        root_node_id = trace_recorder.add_node(
            step_type="user_request",
            parent_id=None,
            model=root_model_name,
            role="user",
            content_preview=user_preview,
        )

        start_time = datetime.now()
        final_response = None
        working_messages = [m.copy() for m in messages]
        resume_state: Optional[Dict[str, Any]] = None
        tool_trace: List[str] = []

        for _ in range(max_tool_iterations):
            final_stage: Dict[str, Any] = {}
            run_optional_params = dict(optional_params)
            run_optional_params["trace_recorder"] = trace_recorder
            run_optional_params["trace_root_node_id"] = root_node_id
            run_optional_params["final_stage"] = final_stage
            run_optional_params["tool_trace_context"] = list(tool_trace)
            if resume_state:
                run_optional_params["resume_state"] = resume_state

            model_override = target_model
            run_litellm_params = dict(safe_kwargs)
            if passthrough_optional_params:
                run_litellm_params.update(passthrough_optional_params)

            final_response = strategy.execute(
                messages=working_messages,
                model_config=model_override,
                optional_params=run_optional_params,
                litellm_params=run_litellm_params,
            )

            if isinstance(final_stage, dict):
                final_stage["model"] = model_override

            message = final_response.choices[0].message
            tool_calls = _extract_tool_calls(message)
            if not tool_calls or not tool_executor:
                break

            resume_messages: List[Dict[str, Any]] = []
            if final_stage and final_stage.get("messages"):
                resume_messages = [m.copy() for m in final_stage["messages"]]
            else:
                resume_messages = [m.copy() for m in working_messages]

            resume_messages.append(
                {
                    "role": "assistant",
                    "content": getattr(message, "content", "") or "",
                    "tool_calls": tool_calls,
                }
            )

            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                tool_trace.append(f"- {tool_name}({arguments})")
                try:
                    result = tool_executor(tool_name, arguments)
                    content = str(result)
                except Exception as exc:
                    content = f"Error executing {tool_name}: {exc}"

                tool_trace.append(f"  -> {content}")

                if trace_recorder:
                    trace_recorder.add_node(
                        step_type="tool_call",
                        parent_id=root_node_id,
                        model=None,
                        role="tool",
                        content_preview=content[:200],
                        extra={"tool_name": tool_name, "arguments": arguments},
                    )

                resume_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": content,
                    }
                )

            if final_stage:
                resume_state = dict(final_stage)
                resume_state["messages"] = [m.copy() for m in resume_messages]
                resume_state["tool_trace"] = list(tool_trace)
                resume_state.setdefault("model", model_override)

            working_messages = resume_messages

        end_time = datetime.now()
        if final_response is not None:
            try:
                log_success(
                    {
                        "model": model,
                        "messages": messages,
                        "trace": trace_recorder.to_dict(),
                    },
                    final_response,
                    start_time,
                    end_time,
                )
            except Exception:
                pass

        return final_response


def register_composite_provider() -> CompositeLiteLLMProvider:
    provider = CompositeLiteLLMProvider()

    custom_providers = getattr(litellm, "_custom_providers", None)
    if custom_providers is None:
        custom_providers = []
        setattr(litellm, "_custom_providers", custom_providers)

    if isinstance(custom_providers, set):
        custom_providers.add("composite")
    elif isinstance(custom_providers, list) and "composite" not in custom_providers:
        custom_providers.append("composite")

    provider_map = getattr(litellm, "custom_provider_map", None)
    if not isinstance(provider_map, list):
        provider_map = []
        setattr(litellm, "custom_provider_map", provider_map)

    provider_list = cast(List[Dict[str, Any]], provider_map)
    existing = next(
        (
            entry
            for entry in provider_list
            if isinstance(entry, dict) and entry.get("provider") == "composite"
        ),
        None,
    )
    if existing is not None:
        existing["custom_handler"] = provider
    else:
        provider_list.append({"provider": "composite", "custom_handler": provider})

    provider_list = getattr(litellm, "provider_list", None)
    if isinstance(provider_list, list) and "composite" not in provider_list:
        provider_list.append("composite")

    return provider
