from __future__ import annotations

import asyncio
from datetime import datetime
import importlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import litellm

from .config_schema import parse_composite_config
from .observability import configure_observability, log_success
from .strategies.base import BaseStrategy
from .trace import TraceRecorder, get_user_request_preview


DEFAULT_STRATEGY_REGISTRY: Dict[str, str] = {
    "moa": "composite_llm.strategies.moa:MoAStrategy",
    "council": "composite_llm.strategies.council:CouncilStrategy",
}


def _import_symbol(import_path: str) -> Any:
    """Import `module:attr` or `module.attr`."""
    if ":" in import_path:
        module_path, attr_path = import_path.split(":", 1)
    else:
        module_path, attr_path = import_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    value: Any = module
    for part in attr_path.split("."):
        value = getattr(value, part)
    return value


def _safe_read_yaml(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}

    try:
        content = Path(path).read_text(encoding="utf-8")
    except Exception:
        return {}

    try:
        data = yaml.safe_load(content)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _extract_composite_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer nesting under general_settings to avoid proxy strictness on unknown top-level keys.
    general = config.get("general_settings")
    if isinstance(general, dict):
        nested = general.get("composite_llm")
        if isinstance(nested, dict):
            return nested
    # Back-compat / alternate layouts
    for key in ("composite_llm", "composite_llm_settings"):
        value = config.get(key)
        if isinstance(value, dict):
            return value
    return {}


def load_composite_provider_config(config_file_path: str | None) -> Dict[str, Any]:
    config = _safe_read_yaml(config_file_path)
    settings = _extract_composite_settings(config)
    strategy_registry = settings.get("strategy_registry")
    observability_settings = settings.get("observability")
    if isinstance(observability_settings, dict):
        configure_observability(observability_settings)
    composite_config = None
    if isinstance(settings, dict):
        composite_config = parse_composite_config(settings)

    return {
        "strategy_registry": strategy_registry
        if isinstance(strategy_registry, dict)
        else {},
        "composite_config": composite_config,
    }


def _parse_composite_model(model: str) -> Tuple[str, str, str, str, Optional[str]]:
    parts = model.split("/")
    if model.startswith("composite/profile/"):
        if len(parts) < 3:
            raise ValueError(
                f"Invalid composite model string: {model}. Expected 'composite/profile/<name>'."
            )
        profile_name = "/".join(parts[2:])
        return "profile", profile_name, model, "composite", profile_name

    if parts and parts[0] == "profile":
        if len(parts) < 2:
            raise ValueError(
                f"Invalid composite model string: {model}. Expected 'profile/<name>'."
            )
        profile_name = "/".join(parts[1:])
        root_model = f"composite/{model}"
        return "profile", profile_name, root_model, "composite", profile_name

    if model.startswith("composite/"):
        if len(parts) < 3:
            raise ValueError(
                f"Invalid composite model string: {model}. Expected 'composite/<strategy>/<model>'."
            )

        strategy_name = parts[1]
        target_model = "/".join(parts[2:])
        root_model = model
        return strategy_name, target_model, root_model, "composite", None

    if parts and parts[0] in {"moa", "council"}:
        if len(parts) < 2:
            raise ValueError(
                f"Invalid composite model string: {model}. Expected '<strategy>/<model>'."
            )
        strategy_name = parts[0]
        target_model = "/".join(parts[1:])
        root_model = f"composite/{model}"
        return strategy_name, target_model, root_model, "composite", None

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


def _resolve_env_vars(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("os.environ/"):
        env_name = value.split("/", 1)[1]
        return os.environ.get(env_name)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


class CompositeLiteLLMProvider:
    def __init__(
        self,
        *,
        strategy_registry: Optional[Dict[str, Any]] = None,
        composite_config: Any = None,
    ):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_registry: Dict[str, Any] = strategy_registry or {}
        self.composite_config = composite_config

    def _resolve_provider_bundle(self, provider_name: str) -> Dict[str, Any]:
        if not self.composite_config:
            return {}
        providers = getattr(self.composite_config, "providers", {}) or {}
        provider = providers.get(provider_name)
        if provider is None:
            raise ValueError(f"Unknown provider bundle: {provider_name}")
        return _resolve_env_vars(cast(Dict[str, Any], provider.litellm_params))

    def _materialize_node(
        self,
        node: Dict[str, Any],
        base_litellm_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        node_model = node.get("model")
        if not isinstance(node_model, str) or not node_model:
            raise ValueError("Node 'model' must be a non-empty string")

        litellm_params: Dict[str, Any] = dict(base_litellm_params)
        provider_name = node.get("provider")
        if isinstance(provider_name, str) and provider_name:
            litellm_params.update(self._resolve_provider_bundle(provider_name))

        node_params = node.get("params")
        if isinstance(node_params, dict):
            litellm_params.update(_resolve_env_vars(node_params))

        return {
            "model": node_model,
            "litellm_params": litellm_params,
        }

    def _get_strategy(self, strategy_name: str) -> BaseStrategy:
        if strategy_name in self.strategies:
            return self.strategies[strategy_name]

        entry: Any = (
            self.strategy_registry.get(strategy_name)
            or DEFAULT_STRATEGY_REGISTRY.get(strategy_name)
        )
        if not entry:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        resolved: Any = entry
        if isinstance(entry, str):
            resolved = _import_symbol(entry)

        new_strategy: BaseStrategy
        if isinstance(resolved, type):
            new_strategy = cast(BaseStrategy, resolved())
        elif callable(resolved):
            new_strategy = cast(BaseStrategy, resolved())
        else:
            raise TypeError(
                f"Invalid strategy registry entry for '{strategy_name}': {entry!r}"
            )

        self.strategies[strategy_name] = new_strategy
        return new_strategy

    async def acompletion(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Any:
        return await asyncio.to_thread(self.completion, model, messages, **kwargs)

    def completion(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Any:
        if not (
            model.startswith("composite/")
            or model.startswith("moa/")
            or model.startswith("council/")
            or model.startswith("profile/")
        ):
            return litellm.completion(model=model, messages=messages, **kwargs)

        strategy_name, target_model, root_model, provider_name, profile_name = _parse_composite_model(
            model
        )
        root_model_name = root_model

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

        if "optional_params" in kwargs:
            kwargs.pop("optional_params")

        allowed_kwargs = {"api_key", "api_base", "timeout", "headers", "max_retries"}
        safe_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}

        profile_optional_params: Dict[str, Any] = {}
        model_override = target_model
        if profile_name:
            composite_config = self.composite_config
            profiles = getattr(composite_config, "profiles", {}) if composite_config else {}
            profile = profiles.get(profile_name)
            if profile is None:
                raise ValueError(f"Unknown composite profile: {profile_name}")

            strategy_name = profile.strategy
            topology = profile.topology
            settings = topology.get("settings", {}) if isinstance(topology, dict) else {}
            if isinstance(settings, dict):
                profile_optional_params.update(settings)

            if profile.strategy == "moa":
                aggregator_node = self._materialize_node(
                    topology["aggregator"], safe_kwargs
                )
                proposers_nodes = [
                    self._materialize_node(node, safe_kwargs)
                    for node in topology.get("proposers", [])
                ]
                model_override = aggregator_node["model"]
                profile_optional_params.update(
                    {
                        "aggregator": aggregator_node,
                        "proposers": proposers_nodes,
                    }
                )
            elif profile.strategy == "council":
                chairman_node = self._materialize_node(
                    topology["chairman"], safe_kwargs
                )
                council_nodes = [
                    self._materialize_node(node, safe_kwargs)
                    for node in topology.get("council", [])
                ]
                reviewers_nodes = [
                    self._materialize_node(node, safe_kwargs)
                    for node in (topology.get("reviewers") or [])
                ]
                model_override = chairman_node["model"]
                profile_optional_params.update(
                    {
                        "chairman": chairman_node,
                        "council": council_nodes,
                    }
                )
                if reviewers_nodes:
                    profile_optional_params["reviewers"] = reviewers_nodes
            else:
                raise ValueError(f"Unsupported profile strategy: {profile.strategy}")

        request_optional_params = optional_params if isinstance(optional_params, dict) else {}
        optional_params = {}
        if profile_optional_params:
            optional_params.update(profile_optional_params)
        optional_params.update(request_optional_params)

        tool_executor = optional_params.get("tool_executor")
        max_tool_iterations = optional_params.get("max_tool_iterations", 5)

        strategy = self._get_strategy(strategy_name)

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


def register_composite_provider(
    *,
    config_file_path: Optional[str] = None,
) -> CompositeLiteLLMProvider:
    """Legacy imperative registration.

    Prefer YAML-first loading via LiteLLM proxy `custom_provider_map`:

    - provider: composite
      custom_handler: composite_llm.litellm_provider:get_composite_provider

    This function remains for non-proxy / direct-Python workflows.
    """

    provider = get_composite_provider(config_file_path=config_file_path)

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

    try:
        from litellm.utils import custom_llm_setup

        custom_llm_setup()
    except Exception:
        pass

    return provider


def get_composite_provider(
    *,
    config_file_path: Optional[str] = None,
    **_: Any,
) -> CompositeLiteLLMProvider:
    """Factory for LiteLLM `custom_provider_map` YAML loading."""
    cfg = load_composite_provider_config(config_file_path)
    return CompositeLiteLLMProvider(
        strategy_registry=cast(Optional[Dict[str, Any]], cfg.get("strategy_registry")),
        composite_config=cfg.get("composite_config"),
    )


def _resolve_env_config_path() -> Optional[str]:
    return os.environ.get("CONFIG_FILE_PATH") or os.environ.get("LITELLM_CONFIG")


def _resolve_proxy_config_path() -> Optional[str]:
    try:
        from litellm.proxy import proxy_server

        return getattr(proxy_server, "user_config_file_path", None)
    except Exception:
        return None


def get_composite_provider_from_env() -> CompositeLiteLLMProvider:
    """Factory that resolves the proxy config path from environment variables."""
    return get_composite_provider(config_file_path=_resolve_env_config_path())


class CompositeProviderProxy:
    """Proxy-safe handler that resolves config path lazily."""

    def __init__(self, config_file_path: Optional[str] = None) -> None:
        self._config_file_path = config_file_path
        self._provider: Optional[CompositeLiteLLMProvider] = None

    def _resolve_config_path(self) -> Optional[str]:
        return self._config_file_path or _resolve_env_config_path() or _resolve_proxy_config_path()

    def _ensure_provider(self) -> CompositeLiteLLMProvider:
        config_path = self._resolve_config_path()
        if self._provider is None or self._config_file_path != config_path:
            self._config_file_path = config_path
            self._provider = get_composite_provider(config_file_path=config_path)
        return self._provider

    async def acompletion(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Any:
        provider = self._ensure_provider()
        return await provider.acompletion(model=model, messages=messages, **kwargs)

    def completion(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Any:
        provider = self._ensure_provider()
        return provider.completion(model=model, messages=messages, **kwargs)


# Proxy-compatible instance (LiteLLM proxy expects a concrete handler object)
COMPOSITE_PROVIDER = CompositeProviderProxy()
