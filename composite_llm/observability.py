import json
import time
import os
from datetime import datetime

LOG_FILE = "llm_logs.jsonl"

def log_success(kwargs, response_obj, start_time, end_time):
    """
    Callback for successful API calls.
    kwargs: dict containing parameters passed to completion()
    response_obj: the ModelResponse object
    """
    try:
        duration = end_time - start_time
        
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

        # Create log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(end_time).isoformat(),
            "status": "success",
            "model": model,
            "duration_seconds": duration,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            # We avoid logging full content for privacy/size in this demo, 
            # but in production you might want it.
            # "input_snippet": str(messages)[:100],
            # "output_snippet": str(response_obj.choices[0].message.content)[:100]
        }
        
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    except Exception as e:
        print(f"Logging error: {e}")

def log_failure(kwargs, exception, start_time, end_time):
    """Callback for failed API calls."""
    try:
        duration = end_time - start_time
        model = kwargs.get("model", "unknown")
        
        log_entry = {
            "timestamp": datetime.fromtimestamp(end_time).isoformat(),
            "status": "failure",
            "model": model,
            "duration_seconds": duration,
            "error": str(exception)
        }
        
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    except Exception as e:
        print(f"Logging error: {e}")

