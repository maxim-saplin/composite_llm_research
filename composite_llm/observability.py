import json
import time
import os
from datetime import datetime
from typing import Any, Dict

LOG_FILE = "llm_logs.jsonl"
_LOGGING_ENABLED = False


def configure_observability(config: Dict[str, Any] | None) -> None:
    global LOG_FILE, _LOGGING_ENABLED
    if not isinstance(config, dict):
        return
    enabled = config.get("enabled")
    if isinstance(enabled, bool):
        _LOGGING_ENABLED = enabled
    log_file = config.get("log_file")
    if isinstance(log_file, str) and log_file.strip():
        LOG_FILE = log_file


def is_logging_enabled() -> bool:
    return _LOGGING_ENABLED


def log_success(kwargs, response_obj, start_time, end_time):
    """
    Callback for successful API calls.
    kwargs: dict containing parameters passed to completion()
    response_obj: the ModelResponse object
    """
    try:
        if not _LOGGING_ENABLED:
            return
        # litellm passes datetime objects or timestamps depending on version/context
        # If they are datetime objects, we can subtract directly to get timedelta
        if isinstance(end_time, datetime) and isinstance(start_time, datetime):
            duration = (end_time - start_time).total_seconds()
            timestamp_str = end_time.isoformat()
        else:
            # Assume timestamps
            duration = end_time - start_time
            timestamp_obj = datetime.fromtimestamp(float(end_time))  # type: ignore[arg-type]
            timestamp_str = timestamp_obj.isoformat()

        # Extract model and input details
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])

        # Calculate tokens (if available in response, usage)
        usage = getattr(response_obj, "usage", {})
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
        else:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

        cost = (
            getattr(response_obj, "cost", None)
            or getattr(response_obj, "response_cost", None)
            or getattr(response_obj, "total_cost", None)
        )
        if cost is None and isinstance(usage, dict):
            cost = usage.get("cost") or usage.get("response_cost") or usage.get("total_cost")
        if cost is None:
            hidden_params = getattr(response_obj, "_hidden_params", None)
            if isinstance(hidden_params, dict):
                cost = hidden_params.get("response_cost") or hidden_params.get("total_cost")
        if cost is not None:
            try:
                cost = float(cost)
            except (TypeError, ValueError):
                cost = None

        # Create log entry
        log_entry = {
            "timestamp": timestamp_str,
            "status": "success",
            "model": model,
            "duration_seconds": duration,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            # We avoid logging full content for privacy/size in this demo,
            # but in production you might want it.
            # "input_snippet": str(messages)[:100],
            # "output_snippet": str(response_obj.choices[0].message.content)[:100]
        }

        # Optional trace graph for composite calls
        trace = kwargs.get("trace")
        if trace:
            # Attach full trace object and top-level trace_id for convenience
            log_entry["trace"] = trace
            if isinstance(trace, dict) and "trace_id" in trace:
                log_entry["trace_id"] = trace["trace_id"]

        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    except Exception as e:
        print(f"Logging error: {e}")


def log_failure(kwargs, exception, start_time, end_time):
    """Callback for failed API calls."""
    try:
        if not _LOGGING_ENABLED:
            return
        if isinstance(end_time, datetime) and isinstance(start_time, datetime):
            duration = (end_time - start_time).total_seconds()
            timestamp_str = end_time.isoformat()
        else:
            duration = end_time - start_time
            timestamp_obj = datetime.fromtimestamp(float(end_time))  # type: ignore[arg-type]
            timestamp_str = timestamp_obj.isoformat()

        model = kwargs.get("model", "unknown")

        log_entry = {
            "timestamp": timestamp_str,
            "status": "failure",
            "model": model,
            "duration_seconds": duration,
            "error": str(exception),
        }

        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    except Exception as e:
        print(f"Logging error: {e}")
