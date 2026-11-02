"""One-round and two-round hybrid RAG over raw chunks (Appendix A.2).

Purpose: the reference system every tax and benefit is measured against.
Called by: pipeline.make_system() when config system.name is rag.
Assumes: chunks come from chunking.chunk_history; corpus embeddings are either the authors'
.npz files (rows aligned with chunk order, D-028) or built once by build() into index_dir.
"""

import json
import os
import re
from pathlib import Path

import numpy as np

from memtax import chunking
from memtax.llm import EMBED_QUERY_PREFIX, Client, Endpoint
from memtax.retrieval import BM25, FloatArray, dense_scores, session_recall, top_k, union
from memtax.schema import Question, Retrieved

# WHY: the paper doesn't publish its sufficiency prompt, so this one is ours (D-030); two-round
# RAG is not gated for that reason.
SUFFICIENCY_PROMPT_VERSION = "memtax/sufficiency-v1"
SUFFICIENCY_PROMPT = (
    "You are checking whether snippets retrieved from past conversations are enough to answer "
    "a question.\n\n"
    "Current date: {date}\n"
    "Question: {question}\n\n"
    "Snippets:\n"
    "{snippets}\n\n"
    "If the snippets contain everything needed to answer, reply "
    '{{"sufficient": true, "why": "<one sentence>", "next_query": ""}}.\n'
    "Otherwise reply "
    '{{"sufficient": false, "why": "<one sentence>", '
    '"next_query": "<a search query for the missing information>"}}.\n'
    "Reply with the JSON object only."
)


class RagSystem:
    def __init__(
        self,
        client: Client,
        *,
        embed: Endpoint,
        rerank: Endpoint,
        chat: Endpoint | None,
        rounds: int,
        bm25_k: int,
        dense_k: int,
        rerank_k: int,
        index_dir: Path,
        authors_embeddings: Path | None,
        dataset_index: dict[str, int],
    ) -> None:
        if rounds not in (1, 2) or (rounds == 2 and chat is None):
            raise ValueError("rag needs rounds 1 or 2, and an answer endpoint for 2 rounds")
        self.client, self.embed_ep, self.rerank_ep, self.chat_ep = client, embed, rerank, chat
        self.rounds, self.bm25_k, self.dense_k, self.rerank_k = rounds, bm25_k, dense_k, rerank_k
        self.index_dir, self.authors_embeddings = index_dir, authors_embeddings
        self.dataset_index = dataset_index

    async def build(self, q: Question) -> None:
        """Embed the history's chunks once and save them; a no-op when reusing the authors'."""
        path = self.index_dir / f"{q.question_id}.npz"
        if self.authors_embeddings is not None or path.exists():
            return
        chunks = chunking.chunk_history(q)
        # WHY: corpus vectors live in .npz files, not the SQLite cache, which they would bloat.
        vectors = await self.client.embed(
            self.embed_ep, [c.text for c in chunks], op="construction", use_cache=False
        )
        self.index_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, embeddings=np.array(vectors, dtype=np.float32))
        os.replace(tmp, path)  # FLOW: atomic, so a crash never leaves a half-written index

    def _matrix(self, q: Question, n_chunks: int) -> FloatArray:
        if self.authors_embeddings is not None:
            i = self.dataset_index[q.question_id]
            path = self.authors_embeddings / f"sample{i}_qwen3emb06b.npz"
        else:
            path = self.index_dir / f"{q.question_id}.npz"
        with np.load(path, allow_pickle=False) as z:
            matrix: FloatArray = z["embeddings"].astype(np.float64)
        if matrix.shape[0] != n_chunks:
            raise ValueError(f"{path}: {matrix.shape[0]} rows for {n_chunks} chunks")
        return matrix

    async def retrieve(self, q: Question) -> Retrieved:
        chunks = chunking.chunk_history(q)
        texts = [c.text for c in chunks]
        matrix, bm25 = self._matrix(q, len(chunks)), BM25(texts)
        picked = await self._round(q.question, texts, matrix, bm25, list(range(len(chunks))))
        round_of, decisions = [1] * len(picked), []
        if self.rounds == 2 and self.chat_ep is not None:
            raw = await self.client.chat(
                self.chat_ep,
                [{"role": "user", "content": _sufficiency_prompt(q, [texts[i] for i in picked])}],
                op="retrieval",
                prompt_version=SUFFICIENCY_PROMPT_VERSION,
                max_tokens=256,
            )
            decisions.append(raw)
            sufficient, next_query = parse_sufficiency(raw, q.question)
            if not sufficient:
                # PAPER: A.2, the follow-up query searches "the remaining pool".
                remaining = [i for i in range(len(chunks)) if i not in set(picked)]
                second = await self._round(next_query, texts, matrix, bm25, remaining)
                picked, round_of = picked + second, round_of + [2] * len(second)
        sessions = [chunks[i].session_id for i in picked]
        return Retrieved(
            q.question_id,
            [chunks[i].doc_id for i in picked],
            [texts[i] for i in picked],
            round_of,
            sessions,
            session_recall(q.answer_session_ids, sessions),
            decisions,
        )

    async def _round(
        self, query: str, texts: list[str], matrix: FloatArray, bm25: BM25, allowed: list[int]
    ) -> list[int]:
        if not allowed:
            return []
        qvec = await self.client.embed(self.embed_ep, [EMBED_QUERY_PREFIX + query], op="retrieval")
        # PAPER: A.2, BM25 and dense each return top-k; the deduplicated union is reranked.
        candidates = union(
            top_k(bm25.scores(query), self.bm25_k, allowed),
            top_k(dense_scores(matrix, np.array(qvec[0])), self.dense_k, allowed),
        )
        scores = await self.client.rerank(
            self.rerank_ep, query, [texts[i] for i in candidates], op="retrieval"
        )
        order = sorted(range(len(candidates)), key=lambda j: -scores[j])
        return [candidates[j] for j in order[: self.rerank_k]]


def _sufficiency_prompt(q: Question, snippets: list[str]) -> str:
    block = "\n\n".join(f"[doc {i}] {s}" for i, s in enumerate(snippets, 1))
    return SUFFICIENCY_PROMPT.format(date=q.question_date, question=q.question, snippets=block)


def parse_sufficiency(raw: str, fallback_query: str) -> tuple[bool, str]:
    """(sufficient, next query). Unparsable replies stop after round one (D-030)."""
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        reply = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        reply = None
    if not isinstance(reply, dict) or not isinstance(reply.get("sufficient"), bool):
        return True, fallback_query
    query = reply.get("next_query")
    return reply["sufficient"], query.strip() if isinstance(
        query, str
    ) and query.strip() else fallback_query
