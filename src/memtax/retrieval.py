"""BM25 and dense scoring over one history's chunks, plus top-k and union helpers.

Purpose: the non-LLM half of RAG retrieval; reranking goes through llm.py.
Called by: systems/rag.py, once per question and round.
Assumes: one history is small (about 450 chunks), so exact scoring needs no index (D-011).
"""

import math
import re
from collections import Counter

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25:
    """Okapi BM25 with Lucene's always-positive idf (D-029)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lengths = np.array([sum(tf.values()) for tf in self.tfs], dtype=np.float64)
        self.avg_len = float(self.lengths.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter(w for tf in self.tfs for w in tf)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> FloatArray:
        out = np.zeros(len(self.tfs), dtype=np.float64)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avg_len or 1.0))
        for w in set(tokenize(query)):
            if w not in self.idf:
                continue
            tf = np.array([d[w] for d in self.tfs], dtype=np.float64)
            out += self.idf[w] * tf * (self.k1 + 1) / (tf + norm)
        return out


def dense_scores(matrix: FloatArray, query: FloatArray) -> FloatArray:
    """Cosine similarity of the query with every row."""
    rows = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    result: FloatArray = rows @ (query / max(float(np.linalg.norm(query)), 1e-12))
    return result


def top_k(scores: FloatArray, k: int, allowed: list[int]) -> list[int]:
    # WHY: a stable sort breaks score ties by chunk order, so runs are reproducible.
    order = sorted(allowed, key=lambda i: -scores[i])
    return order[:k]


def union(*rankings: list[int]) -> list[int]:
    return list(dict.fromkeys(i for ranking in rankings for i in ranking))


def session_recall(answer_session_ids: list[str], retrieved_session_ids: list[str]) -> float | None:
    """Share of gold sessions with at least one retrieved unit; None when there is no gold."""
    # FLOW: logged per question so retrieval misses show up apart from answer errors (G1).
    if not answer_session_ids:
        return None
    hit = set(retrieved_session_ids)
    return sum(s in hit for s in answer_session_ids) / len(answer_session_ids)
