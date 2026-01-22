import json

import requests


def test_fake_server_chat_completion(fake_openai_server) -> None:
    payload = {
        "model": "unit-test-model",
        "messages": [{"role": "user", "content": "Hello there"}],
    }

    response = requests.post(
        f"{fake_openai_server.base_url}/v1/chat/completions",
        json=payload,
        timeout=5,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"].startswith("Response from")
    assert body["usage"]["total_tokens"] > 0


def test_fake_server_tool_call(fake_openai_server) -> None:
    payload = {
        "model": "tool-aggregator",
        "messages": [{"role": "user", "content": "Need tool"}],
    }
    response = requests.post(
        f"{fake_openai_server.base_url}/v1/chat/completions",
        json=payload,
        timeout=5,
    )
    body = response.json()
    tool_calls = body["choices"][0]["message"]["tool_calls"]
    assert tool_calls
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "lookup"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "query": "deterministic"
    }
