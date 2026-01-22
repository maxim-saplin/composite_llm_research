import uuid
from typing import Any, Dict, List, Optional


def get_user_request_preview(messages: List[Dict[str, Any]], max_len: int = 200) -> str:
    """
    Extract a short preview of the user's request from a chat history.
    """
    if not messages:
        return ""

    # Prefer the last user message, fallback to the last message
    user_messages = [m for m in messages if m.get("role") == "user"]
    msg = user_messages[-1] if user_messages else messages[-1]
    content = msg.get("content", "")
    text = str(content)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _get_usage_value(usage: Any, key: str) -> Optional[int]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_usage_metrics(response_obj: Any) -> Dict[str, Optional[float]]:
    if response_obj is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": None,
        }

    usage = getattr(response_obj, "usage", None)
    prompt_tokens = _get_usage_value(usage, "prompt_tokens") or 0
    completion_tokens = _get_usage_value(usage, "completion_tokens") or 0
    total_tokens = _get_usage_value(usage, "total_tokens") or 0

    cost = (
        getattr(response_obj, "cost", None)
        or getattr(response_obj, "response_cost", None)
        or getattr(response_obj, "total_cost", None)
    )
    if cost is None and isinstance(usage, dict):
        cost = (
            usage.get("cost") or usage.get("response_cost") or usage.get("total_cost")
        )
    if cost is None:
        hidden_params = getattr(response_obj, "_hidden_params", None)
        if isinstance(hidden_params, dict):
            cost = hidden_params.get("response_cost") or hidden_params.get("total_cost")

    if cost is not None:
        try:
            cost = float(cost)
        except (TypeError, ValueError):
            cost = None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
    }


class TraceRecorder:
    """
    Lightweight per-request execution graph builder.

    Each composite call (e.g., MoA, Council) can create a
    TraceRecorder and register nodes representing internal steps:

        - user_request
        - strategy
        - llm_call
        - aggregation
        - review
        - chairman
        - tool_call

    The resulting trace can be serialized to a dict and logged to jsonl.
    """

    def __init__(self, strategy: str, root_model: str, user_request_preview: str):
        self.trace_id: str = str(uuid.uuid4())
        self.strategy: str = strategy
        self.root_model: str = root_model
        self.user_request_preview: str = user_request_preview
        self.nodes: List[Dict[str, Any]] = []
        self.root_node_id: Optional[str] = None
        self.aggregate_usage: Dict[str, Optional[float]] = {
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
            "cost": None,
        }

    def add_node(
        self,
        step_type: str,
        parent_id: Optional[str],
        *,
        model: Optional[str] = None,
        role: Optional[str] = None,
        content_preview: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a node to the trace graph.

        Returns the node id, which can be used as parent_id for children.
        """
        node_id = f"n{len(self.nodes)}"

        node: Dict[str, Any] = {
            "id": node_id,
            "parent_id": parent_id,
            "step_type": step_type,
            "strategy": self.strategy,
        }

        if model is not None:
            node["model"] = model
        if role is not None:
            node["role"] = role
        if content_preview is not None:
            node["content_preview"] = content_preview
        if duration_seconds is not None:
            node["duration_seconds"] = duration_seconds

        if prompt_tokens is not None:
            node["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            node["completion_tokens"] = completion_tokens
        if total_tokens is not None:
            node["total_tokens"] = total_tokens
        if cost is not None:
            node["cost"] = cost

        if extra:
            node["extra"] = extra

        self.nodes.append(node)

        if prompt_tokens is not None:
            current_prompt = float(self.aggregate_usage["prompt_tokens"] or 0.0)
            self.aggregate_usage["prompt_tokens"] = current_prompt + float(
                prompt_tokens
            )
        if completion_tokens is not None:
            current_completion = float(self.aggregate_usage["completion_tokens"] or 0.0)
            self.aggregate_usage["completion_tokens"] = current_completion + float(
                completion_tokens
            )
        if total_tokens is not None:
            current_total = float(self.aggregate_usage["total_tokens"] or 0.0)
            self.aggregate_usage["total_tokens"] = current_total + float(total_tokens)
        if cost is not None:
            current_cost = float(self.aggregate_usage["cost"] or 0.0)
            self.aggregate_usage["cost"] = current_cost + float(cost)

        if self.root_node_id is None and parent_id is None:
            self.root_node_id = node_id

        return node_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "strategy": self.strategy,
            "root_model": self.root_model,
            "user_request_preview": self.user_request_preview,
            "nodes": self.nodes,
            "root_node_id": self.root_node_id,
            "aggregate_usage": self.aggregate_usage,
        }
