import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


class FakeOpenAIServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._request_count = 0

    def start(self) -> None:
        handler = self._build_handler()
        self._server = ThreadingHTTPServer((self._host, self._port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._server:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def base_url(self) -> str:
        if not self._server:
            raise RuntimeError("Server is not running.")
        server = self._server
        server_port = server.server_port
        host = self._host
        return f"http://{host}:{server_port}"

    def _build_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - inherited name
                if self.path != "/v1/chat/completions":
                    self.send_response(404)
                    self.end_headers()
                    return

                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                response = outer._build_chat_completion(payload)
                encoded = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: Any) -> None:
                return

        return Handler

    def _build_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._request_count += 1
        model = payload.get("model", "")
        messages = payload.get("messages", []) or []
        tool_calls = []
        content = self._build_content(model, messages)
        has_tool_message = any(m.get("role") == "tool" for m in messages)

        if "tool-aggregator" in model and not has_tool_message:
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": json.dumps({"query": "deterministic"}),
                    },
                }
            ]
            content = ""

        prompt_tokens = self._count_tokens(messages)
        completion_tokens = self._count_tokens([{"content": content}])
        if tool_calls and completion_tokens == 0:
            completion_tokens = 1

        return {
            "id": f"chatcmpl-test-{self._request_count}",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "stop" if not tool_calls else "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _build_content(self, model: str, messages: List[Dict[str, Any]]) -> str:
        tool_message = next(
            (m for m in reversed(messages) if m.get("role") == "tool"), None
        )
        if tool_message:
            tool_text = str(tool_message.get("content", "")).strip()
            return f"Tool-based answer from {model}: {tool_text}"

        user_message = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        user_text = str(user_message.get("content", "")).strip() if user_message else ""
        if "aggregator" in model:
            return f"Aggregated answer from {model}"
        if "chairman" in model:
            return f"Chairman answer from {model}"
        if "reviewer" in model:
            return f"Review from {model} for {user_text}"
        if "proposer" in model:
            return f"Proposal from {model} for {user_text}"
        return f"Response from {model} for {user_text}"

    @staticmethod
    def _count_tokens(messages: List[Dict[str, Any]]) -> int:
        text = " ".join(str(m.get("content", "")) for m in messages if m is not None)
        words = [word for word in text.split() if word]
        return len(words)
