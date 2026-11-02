"""Stand-in for the chat, embedding and rerank servers, so the pipeline runs with no GPU.

Purpose: lets CI and `memtax smoke` exercise every network path in llm.py deterministically.
Called by: tests (test_cache, test_resume, test_pipeline) and cli.smoke(), which loads this
file by path because tests/ is not part of the package (ARCHITECTURE.md).
Assumes: requests follow the shapes llm.py sends; replies are deterministic and meaningless
beyond plumbing: judge says CORRECT when the gold text appears in the response.
"""

import hashlib
import json
import re
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DIM = 32


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _embed(text: str) -> list[float]:
    # WHY: hashed bag of words gives stable vectors where shared words mean higher similarity.
    v = [0.0] * DIM
    for w in _words(text):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


def _chat(messages: list[dict[str, str]]) -> str:
    text = messages[-1]["content"]
    if "Model Response:" in text:  # judge prompt (vendored GENERAL or PREF)
        gold = re.search(r"(?:Gold answer|Rubric): (.*)", text)
        resp = re.search(r"Model Response: (.*)", text)
        ok = gold and resp and gold.group(1).strip().lower() in resp.group(1).lower()
        return '{"label": "CORRECT"}' if ok else '{"label": "WRONG"}'
    if '"sufficient"' in text:  # two-round RAG sufficiency check
        return '{"sufficient": false, "why": "fake", "next_query": "fake follow-up"}'
    # Answer prompt: echo the first retrieved turn, so verdicts vary with retrieval.
    m = re.search(r"\[role: \w+\]: (.*)", text)
    return m.group(1) if m else "The information is not available."


class FakeServer:
    def __init__(self) -> None:
        self.requests: Counter[str] = Counter()
        self.fail_next = 0  # respond 500 to this many requests, to test retries
        self.think = False  # leak "<think>" into chat replies, to test detection
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                route = self.path.rsplit("/", 1)[-1]
                server.requests[route] += 1
                if server.fail_next > 0:
                    server.fail_next -= 1
                    self.send_response(500)
                    self.end_headers()
                    return
                self._reply(server.respond(route, body))

            def _reply(self, payload: Any) -> None:
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def respond(self, route: str, body: dict[str, Any]) -> Any:
        if route == "completions":
            reply = _chat(body["messages"])
            if self.think:
                reply = "<think>hmm</think>" + reply
            prompt = sum(len(_words(m["content"])) for m in body["messages"])
            return {
                "choices": [{"message": {"content": reply}}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": len(_words(reply))},
            }
        if route == "embeddings":
            return {
                "data": [{"index": i, "embedding": _embed(t)} for i, t in enumerate(body["input"])]
            }
        if route == "rerank":
            q = set(_words(body["query"].split("<Query>:")[-1]))
            scores = [len(q & set(_words(d))) / (len(q) or 1) for d in body["documents"]]
            return {"results": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]}
        raise ValueError(f"unknown route {route}")

    def __enter__(self) -> "FakeServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
