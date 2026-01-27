import json
import subprocess
import time
import urllib.request

CONFIG_PATH = "./litellm_proxy.example.yaml"
LOG_PATH = "./composite_proxy_cli_test.log"


def main() -> None:
    with open(LOG_PATH, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [
                "/Users/admin/src/composite_llm_research/.venv/bin/litellm",
                "--config",
                CONFIG_PATH,
                "--host",
                "0.0.0.0",
                "--port",
                "4000",
            ],
            stdout=log,
            stderr=log,
        )

    try:
        time.sleep(3)
        print("--- /v1/models ---")
        with urllib.request.urlopen("http://localhost:4000/v1/models") as resp:
            print(resp.read().decode("utf-8"))

        print("--- /v1/chat/completions (curl) ---")
        payload = json.dumps(
            {
                "model": "composer_cli",
                "messages": [
                    {"role": "user", "content": "Hi! What is your name?"}
                ],
            }
        )
        curl_result = subprocess.run(
            [
                "/usr/bin/curl",
                "-q",
                "-s",
                "-X",
                "POST",
                "http://localhost:4000/v1/chat/completions",
                "-H",
                "Content-Type: application/json",
                "-d",
                payload,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        print(curl_result.stdout.strip())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
