from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class NodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    provider: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class ProviderBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    litellm_params: Dict[str, Any] = Field(default_factory=dict)


class MoATopology(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aggregator: NodeConfig
    proposers: List[NodeConfig]
    settings: Dict[str, Any] = Field(default_factory=dict)


class CouncilTopology(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chairman: NodeConfig
    council: List[NodeConfig]
    reviewers: Optional[List[NodeConfig]] = None
    settings: Dict[str, Any] = Field(default_factory=dict)


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str
    topology: Dict[str, Any]

    @model_validator(mode="after")
    def _validate_topology(self) -> "Profile":
        if self.strategy == "moa":
            self.topology = MoATopology.model_validate(self.topology).model_dump()
        elif self.strategy == "council":
            self.topology = CouncilTopology.model_validate(self.topology).model_dump()
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        return self


class CompositeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    providers: Dict[str, ProviderBundle] = Field(default_factory=dict)
    profiles: Dict[str, Profile] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_provider_refs(self) -> "CompositeConfig":
        provider_names = set(self.providers.keys())
        for profile_name, profile in self.profiles.items():
            topology = profile.topology
            nodes: List[NodeConfig] = []
            if profile.strategy == "moa":
                nodes.append(NodeConfig.model_validate(topology["aggregator"]))
                nodes.extend(
                    NodeConfig.model_validate(node)
                    for node in topology.get("proposers", [])
                )
            elif profile.strategy == "council":
                nodes.append(NodeConfig.model_validate(topology["chairman"]))
                nodes.extend(
                    NodeConfig.model_validate(node)
                    for node in topology.get("council", [])
                )
                nodes.extend(
                    NodeConfig.model_validate(node)
                    for node in (topology.get("reviewers") or [])
                )

            for node in nodes:
                if node.provider and node.provider not in provider_names:
                    raise ValueError(
                        f"Profile '{profile_name}' references unknown provider '{node.provider}'."
                    )

        for provider_name, provider in self.providers.items():
            _validate_env_refs(provider.litellm_params, path=f"providers.{provider_name}")
        return self


def _validate_env_refs(value: Any, path: str) -> None:
    if isinstance(value, str) and value.startswith("os.environ/"):
        env_name = value.split("/", 1)[1]
        if not env_name:
            raise ValueError(f"Invalid env reference at {path}: '{value}'")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_env_refs(item, path=f"{path}.{key}")
    if isinstance(value, list):
        for idx, item in enumerate(value):
            _validate_env_refs(item, path=f"{path}[{idx}]")


def parse_composite_config(raw: Dict[str, Any]) -> Optional[CompositeConfig]:
    if not isinstance(raw, dict):
        return None
    if not any(key in raw for key in ("version", "profiles", "providers")):
        return None
    try:
        return CompositeConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
