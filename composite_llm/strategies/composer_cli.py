"""
Composer CLI Strategy

Invokes the Cursor `agent` CLI directly and treats stdout as the final assistant
response. This strategy bypasses LiteLLM providers entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

from .base import BaseStrategy

ROLE_TOKENS = {
    "system": "<<<ROLE:SYSTEM>>>",
    "user": "<<<ROLE:USER>>>",
    "assistant": "<<<ROLE:ASSISTANT>>>",
}


@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message


class _CliResponse:
    def __init__(self, content: str, *, model: Optional[str] = None) -> None:
        self.choices = [_Choice(_Message(content=content))]
        self.model = model
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


def _escape_role_tokens(text: str) -> str:
    escaped = text
    for token in ROLE_TOKENS.values():
        escaped = escaped.replace(token, f"\\{token}")
    return escaped


def _serialize_transcript(messages: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for msg in messages:
        role = str(msg.get("role", ""))
        token = ROLE_TOKENS.get(role)
        if not token:
            continue
        content = msg.get("content", "")
        if content is None:
            content_text = ""
        else:
            content_text = str(content)
        content_text = _escape_role_tokens(content_text)
        chunks.append(f"{token}\n{content_text}")
    return "\n\n".join(chunks)


def _coerce_timeout_seconds(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _limit_bytes(data: bytes, max_bytes: int) -> tuple[bytes, bool]:
    if max_bytes <= 0:
        return data, False
    if len(data) <= max_bytes:
        return data, False
    return data[:max_bytes], True


def _resolve_io_log_path(optional_params: Dict[str, Any]) -> Optional[str]:
    if not isinstance(optional_params, dict):
        return None
    direct = optional_params.get("io_log_path")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    io_dir = optional_params.get("io_log_dir")
    if isinstance(io_dir, str) and io_dir.strip():
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return str(Path(io_dir).expanduser() / f"composer_cli_{ts}.log")
    return None


def _write_cli_io_log(
    path: str,
    *,
    prompt: str,
    stdout_text: str,
    stderr_text: str,
    meta: Dict[str, Any],
) -> None:
    try:
        log_path = Path(path).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().isoformat() + "Z"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("=== Composer CLI IO ===\n")
            handle.write(f"timestamp: {timestamp}\n")
            for key, value in meta.items():
                handle.write(f"{key}: {value}\n")
            handle.write("\n--- PROMPT ---\n")
            handle.write(prompt)
            handle.write("\n\n--- STDOUT ---\n")
            handle.write(stdout_text)
            handle.write("\n\n--- STDERR ---\n")
            handle.write(stderr_text)
            handle.write("\n\n")
    except Exception:
        pass


class ComposerCliStrategy(BaseStrategy):
    """
    Strategy that shells out to the Cursor `agent` CLI.

    Usage:
        model="composite/composer-cli/<model>"
    """

    def execute(
        self,
        messages: List[Dict[str, str]],
        model_config: str,
        optional_params: Dict[str, Any],
        litellm_params: Dict[str, Any],
    ) -> Any:
        model_name = (
            optional_params.get("model")
            if isinstance(optional_params, dict)
            else None
        ) or model_config
        if not model_name:
            raise ValueError("Composer CLI strategy requires a model name.")

        trace_recorder = optional_params.get("trace_recorder")
        trace_root_id = optional_params.get("trace_root_node_id")
        strategy_node_id = None
        if trace_recorder and trace_root_id is not None:
            strategy_node_id = trace_recorder.add_node(
                step_type="strategy",
                parent_id=trace_root_id,
                model=model_name,
                content_preview="Composer CLI strategy execution",
            )

        resume_state = optional_params.get("resume_state")
        if isinstance(resume_state, dict) and resume_state.get("messages"):
            messages = resume_state.get("messages")  # type: ignore[assignment]

        prompt = _serialize_transcript(messages)
        io_log_path = _resolve_io_log_path(optional_params)

        timeout_seconds = _coerce_timeout_seconds(
            optional_params.get("timeout_seconds"),
            _coerce_timeout_seconds(litellm_params.get("timeout"), 60.0),
        )
        max_output_bytes = optional_params.get("max_output_bytes", 65536)
        if not isinstance(max_output_bytes, int):
            max_output_bytes = 65536

        cli_command = optional_params.get("cli_command", "agent")
        if not isinstance(cli_command, str) or not cli_command.strip():
            cli_command = "agent"
        cli_command = cli_command.strip()
        resolved_cli = None
        cli_path = Path(cli_command).expanduser()
        if cli_path.is_file():
            resolved_cli = str(cli_path.resolve())
        if not resolved_cli and (cli_path.is_absolute() or "/" in cli_command or "\\" in cli_command):
            config_file_path = optional_params.get("config_file_path")
            if not config_file_path:
                config_file_path = os.environ.get("LITELLM_CONFIG") or os.environ.get(
                    "CONFIG_FILE_PATH"
                )
            if isinstance(config_file_path, str) and config_file_path:
                try:
                    config_dir = Path(config_file_path).expanduser().resolve().parent
                    candidate = (config_dir / cli_path).resolve()
                except Exception:
                    candidate = None
                if candidate and candidate.is_file():
                    resolved_cli = str(candidate)
            if not resolved_cli:
                try:
                    repo_root = Path(__file__).resolve().parents[2]
                    candidate = (repo_root / cli_path).resolve()
                except Exception:
                    candidate = None
                if candidate and candidate.is_file():
                    resolved_cli = str(candidate)
        if not resolved_cli:
            resolved_cli = shutil.which(cli_command)
            if resolved_cli:
                resolved_cli = str(Path(resolved_cli).resolve())
        if not resolved_cli:
            if io_log_path:
                _write_cli_io_log(
                    io_log_path,
                    prompt=prompt,
                    stdout_text="",
                    stderr_text=f"Command not found: {cli_command}",
                    meta={
                        "model": model_name,
                        "command": f"{cli_command} -p <prompt> --model {model_name} --mode=ask",
                        "timeout": False,
                        "exit_code": None,
                    },
                )
            if trace_recorder:
                trace_recorder.add_node(
                    step_type="llm_call",
                    parent_id=strategy_node_id,
                    model=model_name,
                    role="assistant",
                    content_preview="CLI command not found",
                    extra={
                        "backend": "composer-cli",
                        "exit_code": None,
                        "stderr": f"Command not found: {cli_command}",
                        "timeout": False,
                        "stdout_truncated": False,
                        "stderr_truncated": False,
                    },
                )
            raise FileNotFoundError(
                f"Composer CLI command not found: '{cli_command}'."
            )

        command = [
            resolved_cli,
            "-p",
            prompt,
            "--model",
            str(model_name),
            "--mode=ask",
        ]
        command_preview = f"{resolved_cli} -p <prompt> --model {model_name} --mode=ask"

        start = time.time()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=False,
                    timeout=timeout_seconds,
                    check=False,
                    cwd=temp_dir,
                )
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - start
            stdout_bytes = exc.stdout or b""
            stderr_bytes = exc.stderr or b""
            stdout_full = stdout_bytes.decode("utf-8", errors="replace")
            stderr_full = stderr_bytes.decode("utf-8", errors="replace")
            stdout_limited, stdout_truncated = _limit_bytes(
                stdout_bytes, max_output_bytes
            )
            stderr_limited, stderr_truncated = _limit_bytes(
                stderr_bytes, max_output_bytes
            )
            stdout_text = stdout_limited.decode("utf-8", errors="replace")
            stderr_text = stderr_limited.decode("utf-8", errors="replace")
            if trace_recorder:
                trace_recorder.add_node(
                    step_type="llm_call",
                    parent_id=strategy_node_id,
                    model=model_name,
                    role="assistant",
                    content_preview=(stdout_text or stderr_text)[:200],
                    duration_seconds=duration,
                    extra={
                        "backend": "composer-cli",
                        "exit_code": None,
                        "stderr": stderr_text[:200],
                        "timeout": True,
                        "stdout_truncated": stdout_truncated,
                        "stderr_truncated": stderr_truncated,
                    },
                )
            if io_log_path:
                _write_cli_io_log(
                    io_log_path,
                    prompt=prompt,
                    stdout_text=stdout_full,
                    stderr_text=stderr_full,
                    meta={
                        "model": model_name,
                        "command": command_preview,
                        "timeout": True,
                        "duration_seconds": round(duration, 3),
                        "exit_code": None,
                    },
                )
            raise TimeoutError(
                f"Composer CLI timed out after {timeout_seconds}s."
            ) from exc

        duration = time.time() - start
        stdout_bytes = result.stdout or b""
        stderr_bytes = result.stderr or b""
        stdout_full = stdout_bytes.decode("utf-8", errors="replace")
        stderr_full = stderr_bytes.decode("utf-8", errors="replace")

        stdout_limited, stdout_truncated = _limit_bytes(stdout_bytes, max_output_bytes)
        stderr_limited, stderr_truncated = _limit_bytes(stderr_bytes, max_output_bytes)
        stdout_text = stdout_limited.decode("utf-8", errors="replace")
        stderr_text = stderr_limited.decode("utf-8", errors="replace")

        if io_log_path:
            _write_cli_io_log(
                io_log_path,
                prompt=prompt,
                stdout_text=stdout_full,
                stderr_text=stderr_full,
                meta={
                    "model": model_name,
                    "command": command_preview,
                    "timeout": False,
                    "duration_seconds": round(duration, 3),
                    "exit_code": result.returncode,
                },
            )

        if trace_recorder:
            trace_recorder.add_node(
                step_type="llm_call",
                parent_id=strategy_node_id,
                model=model_name,
                role="assistant",
                content_preview=(stdout_text or stderr_text)[:200],
                duration_seconds=duration,
                extra={
                    "backend": "composer-cli",
                    "exit_code": result.returncode,
                    "stderr": stderr_text[:200],
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                },
            )

        if result.returncode != 0:
            raise RuntimeError(
                "Composer CLI failed with exit code "
                f"{result.returncode}. stderr: {stderr_text.strip()}"
            )

        content = stdout_text.rstrip()
        return _CliResponse(content=content, model=model_name)
