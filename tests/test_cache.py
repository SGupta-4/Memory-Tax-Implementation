import asyncio
import sqlite3
from pathlib import Path

import pytest

from memtax.llm import Cache, Client, Endpoint, LLMError
from tests.fake_server import FakeServer

MSG = [{"role": "user", "content": "hello there"}]


def _ep(server: FakeServer, revision: str = "r1") -> Endpoint:
    return Endpoint(server.base_url, "fake-model", revision)


def test_second_identical_call_makes_no_network_request(tmp_path: Path) -> None:
    with FakeServer() as server:
        client = Client(Cache(tmp_path / "cache.sqlite"))
        first = asyncio.run(
            client.chat(_ep(server), MSG, op="qa", prompt_version="v", max_tokens=8)
        )
        second = asyncio.run(
            client.chat(_ep(server), MSG, op="qa", prompt_version="v", max_tokens=8)
        )
        assert first == second
        assert server.requests["completions"] == 1
        # Usage still counts both calls, so cost tables don't depend on cache state.
        assert client.usage["qa"]["llm_calls"] == 2
        assert client.usage["qa"]["cache_hits"] == 1


def test_cache_survives_a_new_client(tmp_path: Path) -> None:
    with FakeServer() as server:
        asyncio.run(
            Client(Cache(tmp_path / "c.sqlite")).chat(
                _ep(server), MSG, op="qa", prompt_version="v", max_tokens=8
            )
        )
        asyncio.run(
            Client(Cache(tmp_path / "c.sqlite")).chat(
                _ep(server), MSG, op="qa", prompt_version="v", max_tokens=8
            )
        )
        assert server.requests["completions"] == 1
    integrity = sqlite3.connect(tmp_path / "c.sqlite").execute("PRAGMA integrity_check").fetchone()
    assert integrity == ("ok",)


@pytest.mark.parametrize(
    "change", [{"revision": "r2"}, {"prompt_version": "v2"}, {"max_tokens": 9}, {"attempt": 1}]
)
def test_key_changes_with_revision_prompt_settings_and_attempt(
    tmp_path: Path, change: dict[str, object]
) -> None:
    with FakeServer() as server:
        client = Client(Cache(tmp_path / "c.sqlite"))
        base: dict[str, object] = {"prompt_version": "v", "max_tokens": 8, "attempt": 0}

        def call(revision: str = "r1", **kw: object) -> None:
            args = {**base, **kw}
            asyncio.run(client.chat(_ep(server, revision), MSG, op="qa", **args))  # type: ignore[arg-type]

        call()
        call(**change)
        assert server.requests["completions"] == 2


def test_retries_then_counts_failure(tmp_path: Path) -> None:
    with FakeServer() as server:
        client = Client(Cache(tmp_path / "c.sqlite"), retries=1, backoff=0.0)
        server.fail_next = 1
        asyncio.run(client.chat(_ep(server), MSG, op="qa", prompt_version="v", max_tokens=8))
        assert client.usage["qa"]["failed_calls"] == 0
        server.fail_next = 5
        with pytest.raises(LLMError):
            asyncio.run(client.chat(_ep(server), MSG, op="qa", prompt_version="w", max_tokens=8))
        assert client.usage["qa"]["failed_calls"] == 1


def test_embed_batches_and_rerank_order(tmp_path: Path) -> None:
    with FakeServer() as server:
        client = Client(Cache(tmp_path / "c.sqlite"))
        vectors = asyncio.run(client.embed(_ep(server), ["a b"] * 70, op="index"))
        assert len(vectors) == 70 and server.requests["embeddings"] == 2  # batches of 64
        scores = asyncio.run(client.rerank(_ep(server), "red apple", ["blue", "red apple"], op="r"))
        assert scores[1] > scores[0]
