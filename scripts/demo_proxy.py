from __future__ import annotations

import argparse
import json
import textwrap
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def _format_block(text: str, width: int = 78) -> str:
    lines = text.split("\n")
    formatted: List[str] = []
    for line in lines:
        if line.strip():
            formatted.append(textwrap.fill(line, width=width, subsequent_indent="  "))
        else:
            formatted.append("")
    return "\n".join(formatted)


def _get_json(url: str) -> Tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return int(getattr(e, "code", 0) or 0), e.read().decode("utf-8", errors="replace")


def _post_json(url: str, payload: Dict[str, Any]) -> Tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer local-test",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return int(getattr(e, "code", 0) or 0), e.read().decode("utf-8", errors="replace")


def _parse_models(text: str) -> List[str]:
    try:
        payload = json.loads(text)
    except Exception:
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    models: List[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            models.append(item["id"])
    return models


def _extract_message(payload: Dict[str, Any]) -> str:
    try:
        return str(payload["choices"][0]["message"].get("content", "") or "")
    except Exception:
        return ""


def _extract_reasoning(payload: Dict[str, Any]) -> str:
    try:
        return str(payload["choices"][0]["message"].get("reasoning_content", "") or "")
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretty proxy demo for composite models")
    parser.add_argument("--base-url", default="http://localhost:4000", help="Proxy base URL")
    parser.add_argument(
        "--models",
        default="moa_light,moa_hard,council_basic",
        help="Comma-separated proxy model names",
    )
    parser.add_argument(
        "--prompt",
        default="How many r's in strawberry?",
        help="User prompt",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Hide reasoning_content if present",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    show_reasoning = not args.no_reasoning
    models_url = base_url + "/v1/models"
    chat_url = base_url + "/v1/chat/completions"

    rule = "═" * 78
    print(f"{Colors.BOLD}{Colors.HEADER}{rule}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}Proxy Demo{Colors.RESET}")
    print(f"{Colors.DIM}Base URL: {base_url}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}{rule}{Colors.RESET}\n")

    status, text = _get_json(models_url)
    discovered_models = _parse_models(text)

    if status == 200 and discovered_models:
        print(f"{Colors.GREEN}✓ Models discovered from /v1/models:{Colors.RESET}")
        for model in discovered_models:
            print(f"  - {model}")
    elif status == 200:
        print(f"{Colors.YELLOW}⚠ /v1/models returned an empty list.{Colors.RESET}")
    else:
        print(f"{Colors.RED}✗ Failed to fetch /v1/models (HTTP {status}).{Colors.RESET}")
        print(_format_block(text))

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_list:
        model_list = discovered_models

    print(f"\n{Colors.BOLD}{Colors.HEADER}Requests{Colors.RESET}")
    print(f"{Colors.DIM}Prompt: {args.prompt}{Colors.RESET}\n")

    results: List[Dict[str, Any]] = []
    for model in model_list:
        print(f"{Colors.BOLD}{Colors.CYAN}→ {model}{Colors.RESET}")
        start = time.time()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": args.prompt}],
            "stream": False,
        }
        status, text = _post_json(chat_url, payload)
        elapsed = time.time() - start

        if status != 200:
            print(f"{Colors.RED}✗ HTTP {status}{Colors.RESET}")
            print(_format_block(text))
            results.append({"model": model, "status": status, "ok": False})
            print()
            continue

        try:
            data = json.loads(text)
        except Exception:
            print(f"{Colors.RED}✗ Invalid JSON response{Colors.RESET}")
            print(_format_block(text))
            results.append({"model": model, "status": status, "ok": False})
            print()
            continue

        content = _extract_message(data)
        reasoning = _extract_reasoning(data)
        print(f"{Colors.GREEN}✓ HTTP {status}{Colors.RESET} ({elapsed:.2f}s)")
        if content:
            print(_format_block(content))
        else:
            print(f"{Colors.YELLOW}⚠ Empty content returned.{Colors.RESET}")
        if show_reasoning and reasoning:
            print(f"\n{Colors.DIM}Reasoning:{Colors.RESET}")
            print(_format_block(reasoning))
        results.append(
            {
                "model": model,
                "status": status,
                "ok": True,
                "content": content,
                "reasoning": reasoning,
            }
        )
        print()

    print(f"\n{Colors.BOLD}{Colors.HEADER}{rule}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}📊 SUMMARY{Colors.RESET}")
    print(f"{Colors.HEADER}{rule}{Colors.RESET}")
    for item in results:
        status_icon = f"{Colors.GREEN}✓{Colors.RESET}" if item["ok"] else f"{Colors.RED}✗{Colors.RESET}"
        if item.get("ok") and item.get("content"):
            answer = str(item["content"]).strip().split("\n")[-1]
            if len(answer) > 100:
                answer = "..." + answer[-100:]
            print(
                f"  {status_icon} {item['model']} (HTTP {item['status']}) → {answer}"
            )
        else:
            print(f"  {status_icon} {item['model']} (HTTP {item['status']})")

    return 0 if all(item["ok"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
