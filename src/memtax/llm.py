"""Cached, counted, bounded-concurrency clients for chat, embedding and rerank endpoints.

Purpose: the only path to a model, so every call is cached (D-010), counted per operation for
the cost tables, and retried; failed calls are counted for G1's failure-rate check.
Called by: systems/rag.py (embed, rerank, sufficiency chat), answer.py, judge.py, pipeline.py.
Assumes: OpenAI-compatible /chat/completions and /embeddings, and vLLM's /rerank, as started by
scripts/kaggle_serve.sh; replies are untrusted and validated here. API keys come only from the
environment variable an Endpoint names, and are never logged or cached.
"""

import asyncio
import hashlib
import json
import os
import sqlite3
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

# PAPER: Qwen3-Embedding's query prompt; the authors' query file says query_prompt=true.
EMBED_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)
# PAPER: Qwen3-Reranker model card format; vLLM scores the "yes" token of this template.
_RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
EMBED_BATCH = 64


@dataclass(frozen=True)
class Endpoint:
    base_url: str  # e.g. http://localhost:8000/v1
    model: str  # served model name
    revision: str  # HF weights revision; in the cache key so a new revision never hits old replies
    api_key_env: str = ""  # name of the env var holding the key; empty when none is needed


class LLMError(RuntimeError):
    """A call that still failed after retries, or returned a malformed reply."""


class Cache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        # WHY: WAL keeps committed rows safe if the session dies mid-run (D-010).
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key: str) -> Any:
        row = self._db.execute("SELECT value FROM calls WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, value: Any) -> None:
        self._db.execute("INSERT OR REPLACE INTO calls VALUES (?, ?)", (key, json.dumps(value)))
        self._db.commit()


def cache_key(kind: str, ep: Endpoint, prompt_version: str, payload: dict[str, Any]) -> str:
    # FLOW: the key covers everything that can change a reply; FLOW.md "Each answer and judge call".
    blob = json.dumps(
        {"kind": kind, "model": ep.model, "revision": ep.revision, "pv": prompt_version, **payload},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Client:
    def __init__(
        self,
        cache: Cache | None,
        concurrency: int = 8,
        retries: int = 3,
        timeout: float = 300.0,
        backoff: float = 2.0,
    ) -> None:
        self.cache = cache
        self.retries, self.timeout, self.backoff = retries, timeout, backoff
        self._sem = asyncio.Semaphore(concurrency)
        # FLOW: pipeline.py copies this into the manifest; keys are operation names.
        self.usage: defaultdict[str, Counter[str]] = defaultdict(Counter)

    async def chat(
        self,
        ep: Endpoint,
        messages: list[dict[str, str]],
        *,
        op: str,
        prompt_version: str,
        max_tokens: int,
        attempt: int = 0,
        use_cache: bool = True,
    ) -> str:
        payload = {
            "model": ep.model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            # PAPER: A.4, reasoning disabled. kaggle_serve.sh health checks no <think> leaks.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        # WHY: attempt is only in the key, so a judge retry is a new cached call, not a hit.
        key = cache_key("chat", ep, prompt_version, {"payload": payload, "attempt": attempt})
        reply = await self._cached(key, use_cache, op, lambda: self._chat_raw(ep, payload, op))
        usage = self.usage[op]
        usage["llm_calls"] += 1
        usage["prompt_tokens"] += int(reply["prompt_tokens"])
        usage["completion_tokens"] += int(reply["completion_tokens"])
        return str(reply["text"])

    async def _chat_raw(self, ep: Endpoint, payload: dict[str, Any], op: str) -> dict[str, Any]:
        body = await self._post(ep, "/chat/completions", payload, op)
        try:
            text = body["choices"][0]["message"]["content"]
            usage = body.get("usage") or {}
            if not isinstance(text, str):
                raise TypeError("content is not text")
            return {
                "text": text,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            }
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LLMError(f"malformed chat reply from {ep.base_url}: {e}") from e

    async def embed(
        self, ep: Endpoint, texts: list[str], *, op: str, use_cache: bool = True
    ) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            payload = {"model": ep.model, "input": batch}
            key = cache_key("embed", ep, "", payload)
            # WHY: partial binds this batch's payload now, not the loop variable later.
            vectors = await self._cached(
                key, use_cache, op, partial(self._embed_raw, ep, payload, op)
            )
            out.extend(vectors)
        self.usage[op]["embed_calls"] += len(texts)
        self.usage[op]["embed_chars"] += sum(len(t) for t in texts)
        return out

    async def _embed_raw(self, ep: Endpoint, payload: dict[str, Any], op: str) -> list[list[float]]:
        body = await self._post(ep, "/embeddings", payload, op)
        try:
            rows = sorted(body["data"], key=lambda r: int(r["index"]))
            vectors = [[float(x) for x in r["embedding"]] for r in rows]
        except (KeyError, TypeError, ValueError) as e:
            raise LLMError(f"malformed embedding reply from {ep.base_url}: {e}") from e
        if len(vectors) != len(payload["input"]):
            raise LLMError(
                f"{ep.base_url} returned {len(vectors)} embeddings for {len(payload['input'])}"
            )
        return vectors

    async def rerank(
        self, ep: Endpoint, query: str, documents: list[str], *, op: str
    ) -> list[float]:
        """Relevance score per document, in the order given."""
        payload = {
            "model": ep.model,
            "query": f"{_RERANK_PREFIX}<Instruct>: {_RERANK_INSTRUCTION}\n<Query>: {query}\n",
            "documents": [f"<Document>: {d}{_RERANK_SUFFIX}" for d in documents],
        }
        key = cache_key("rerank", ep, "", payload)
        scores = await self._cached(key, True, op, lambda: self._rerank_raw(ep, payload, op))
        self.usage[op]["rerank_calls"] += 1
        self.usage[op]["rerank_pairs"] += len(documents)
        return [float(s) for s in scores]

    async def _rerank_raw(self, ep: Endpoint, payload: dict[str, Any], op: str) -> list[float]:
        body = await self._post(ep, "/rerank", payload, op)
        n = len(payload["documents"])
        scores: list[float | None] = [None] * n
        try:
            for r in body["results"]:
                scores[int(r["index"])] = float(r["relevance_score"])
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise LLMError(f"malformed rerank reply from {ep.base_url}: {e}") from e
        if any(s is None for s in scores):
            raise LLMError(f"{ep.base_url} scored fewer documents than sent")
        return [s for s in scores if s is not None]

    async def _cached(self, key: str, use_cache: bool, op: str, call: Any) -> Any:
        if use_cache and self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.usage[op]["cache_hits"] += 1
                return hit
        value = await call()
        if self.cache is not None:
            self.cache.put(key, value)
        return value

    async def _post(self, ep: Endpoint, path: str, payload: dict[str, Any], op: str) -> Any:
        url = ep.base_url.rstrip("/") + path
        headers = {"Content-Type": "application/json"}
        if ep.api_key_env:
            headers["Authorization"] = f"Bearer {os.environ.get(ep.api_key_env, '')}"
        data = json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                async with self._sem:
                    self.usage[op]["network_calls"] += 1
                    return await asyncio.to_thread(self._send, url, data, headers)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
                last = e
                if attempt < self.retries:
                    await asyncio.sleep(self.backoff * 2**attempt)
        self.usage[op]["failed_calls"] += 1
        # WHY: the message names the URL and error only; headers (and so keys) never appear.
        raise LLMError(f"{url} failed after {self.retries + 1} attempts: {last!r}")

    def _send(self, url: str, data: bytes, headers: dict[str, str]) -> Any:
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
