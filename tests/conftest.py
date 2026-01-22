import os
from typing import Generator

import pytest

from tests.fake_openai_server import FakeOpenAIServer


@pytest.fixture(scope="session")
def fake_openai_server() -> "Generator[FakeOpenAIServer, None, None]":
    server = FakeOpenAIServer()
    server.start()
    yield server
    server.stop()


@pytest.fixture(autouse=True)
def _fake_openai_env(fake_openai_server: FakeOpenAIServer) -> None:
    os.environ["OPENAI_API_BASE"] = f"{fake_openai_server.base_url}/v1"
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["LITELLM_API_KEY"] = "test-key"
    os.environ["OPENAI_API_TYPE"] = "openai"
    os.environ.pop("AZURE_OPENAI_API_KEY", None)
