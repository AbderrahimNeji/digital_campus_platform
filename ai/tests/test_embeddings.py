"""Gemini embeddings stay opt-in, 768-dimensional, and free of PyTorch."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest

from ai.rag.embeddings import (
    EMBEDDING_DIM,
    _gemini_client,
    embed_query,
    embed_texts,
    embedding_provider,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_dotenv_or_cached_clients(monkeypatch):
    monkeypatch.setattr("ai.rag.embeddings.load_dotenv", lambda *args, **kwargs: False)
    embed_query.cache_clear()
    _gemini_client.cache_clear()
    yield
    embed_query.cache_clear()
    _gemini_client.cache_clear()


def test_provider_defaults_to_local(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "present-but-not-a-switch")
    assert embedding_provider() == "local"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    with pytest.raises(ValueError, match="local"):
        embedding_provider()


def test_gemini_embeddings_are_768_and_normalized(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")
    seen = {}

    class Config:
        def __init__(self, **kwargs):
            seen["config"] = kwargs
            self.__dict__.update(kwargs)

    class Embedding:
        values = [3.0, *([0.0] * (EMBEDDING_DIM - 1))]

    class Result:
        def __init__(self, count):
            self.embeddings = [Embedding() for _ in range(count)]

    class Models:
        def embed_content(self, model, contents, config):
            seen["model"] = model
            seen["contents"] = list(contents)
            seen["task_type"] = config.task_type if hasattr(config, "task_type") else seen["config"]["task_type"]
            return Result(len(contents))

    class Client:
        def __init__(self, api_key):
            assert api_key == "unit-test-key"
            self.models = Models()

    _install_fake_genai(monkeypatch, Client, Config)

    vectors = embed_texts(["bonjour", "مرحبا"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 768
    assert vectors[0][0] == pytest.approx(1.0)
    assert math.isclose(math.sqrt(sum(value * value for value in vectors[0])), 1.0, abs_tol=1e-9)
    assert seen["model"] == "gemini-embedding-001"
    assert seen["config"]["output_dimensionality"] == 768
    assert seen["config"]["task_type"] == "RETRIEVAL_DOCUMENT"

    query = embed_query("question")
    assert len(query) == 768
    assert seen["config"]["task_type"] == "RETRIEVAL_QUERY"


def test_gemini_does_not_load_the_local_model(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")

    def fail_local():
        raise AssertionError("local embedding model was loaded")

    monkeypatch.setattr("ai.rag.embeddings._get_model", fail_local)

    class Config:
        def __init__(self, **kwargs):
            pass

    class Embedding:
        values = [1.0, *([0.0] * (EMBEDDING_DIM - 1))]

    class Result:
        embeddings = [Embedding()]

    class Models:
        def embed_content(self, model, contents, config):
            return Result()

    class Client:
        def __init__(self, api_key):
            self.models = Models()

    _install_fake_genai(monkeypatch, Client, Config)

    assert len(embed_texts(["une phrase"])) == 1


def test_gemini_refuses_a_vector_that_is_not_768(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")

    class Config:
        def __init__(self, **kwargs):
            pass

    class Embedding:
        values = [1.0, 0.0, 0.0]

    class Result:
        embeddings = [Embedding()]

    class Models:
        def embed_content(self, model, contents, config):
            return Result()

    class Client:
        def __init__(self, api_key):
            self.models = Models()

    _install_fake_genai(monkeypatch, Client, Config)

    with pytest.raises(RuntimeError, match="768"):
        embed_texts(["trop court"])


def test_gemini_without_a_key_does_not_call_the_api(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        embed_texts(["bonjour"])


def _install_fake_genai(monkeypatch, client_cls, config_cls):
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = client_cls
    fake_genai.types = types.SimpleNamespace(EmbedContentConfig=config_cls)
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)


def _requirement_names(path: Path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        lines.append(line.split("#", 1)[0].strip().lower())
    return "\n".join(line for line in lines if line)


def test_production_requirements_omit_pytorch():
    prod = _requirement_names(ROOT / "ai" / "requirements-prod.txt")
    local = _requirement_names(ROOT / "ai" / "requirements.txt")
    assert "torch" not in prod
    assert "sentence-transformers" not in prod
    assert "torch" in local
    assert "sentence-transformers" in local

    start = (ROOT / "scripts" / "start_api.sh").read_text(encoding="utf-8")
    assert "--host 0.0.0.0" in start
    assert "PORT" in start
