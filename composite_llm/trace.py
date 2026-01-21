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


class TraceRecorder:
    """
    Lightweight per-request execution graph builder.

    Each composite call (e.g., MoA, Council, CoT, ThinkTool) can create a
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

        if extra:
            node["extra"] = extra

        self.nodes.append(node)

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
        }



