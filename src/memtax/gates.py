"""Phase 1 checks: chunks, embeddings, judge agreement, and gate G1.

Purpose: turns runs and the authors' files into the G1 criteria in configs/gates.yaml, so the
gate is a command, not a judgement call made after seeing results.
Called by: cli check-chunks, check-embeddings, judge-agreement, and gate.
Assumes: thresholds in configs/gates.yaml were set before any run (preregistered); authors'
files are under data_dir in the layout FLOW.md gives.
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from memtax import chunking, data, metrics, pipeline
from memtax.judge import judge_id
from memtax.llm import EMBED_QUERY_PREFIX, Cache, Client
from memtax.schema import Question

GATES = Path("configs/gates.yaml")
QUESTIONS = "longmemeval/longmemeval_s.json"
AUTHORS_RAG1 = "authors/naive_baselines/res_rag_hybrid_1rd_longmemeval_s_4b"
AUTHORS_RAG1_VERDICTS = f"{AUTHORS_RAG1}/judge_res/judge.jsonl"
AUTHORS_RAG1_ANSWERS = f"{AUTHORS_RAG1}/qa_res/qa_results.jsonl"
AUTHOR_CHUNKS = (
    "res_other/preprocess_chunks/main_exp_bench/longmemeval_s/buffer/longmemeval_s_corpus.json"
)
AUTHOR_EMBEDDINGS = "res_other/preprocess_embeddings/main_exp_bench/longmemeval_s"
# PAPER: D-005, reuse the authors' embeddings only if ours match at cosine >= 0.999.
EMBED_COSINE_MIN = 0.999


def thresholds() -> dict[str, Any]:
    g1: dict[str, Any] = yaml.safe_load(GATES.read_text(encoding="utf-8"))["g1"]
    return g1


def chunk_match(data_dir: Path) -> tuple[int, int, list[str]]:
    """(matching histories, total, first mismatched ids)."""
    questions = data.load_questions(data_dir / QUESTIONS)
    theirs = data.load_author_chunks(data_dir / AUTHOR_CHUNKS)
    bad = [
        q.question_id
        for q in questions
        if [(c.doc_id, c.text, c.turn_ids) for c in chunking.chunk_history(q)]
        != theirs.get(q.question_id)
    ]
    return len(questions) - len(bad), len(questions), bad[:5]


def _dev100(cfg: pipeline.Config) -> tuple[list[Question], dict[str, int]]:
    questions = data.load_questions(cfg.questions)
    index = {q.question_id: i for i, q in enumerate(questions)}
    return pipeline.select_questions(questions, "dev100"), index


async def check_embeddings(cfg: pipeline.Config, data_dir: Path) -> dict[str, float]:
    """Cosine of our dev-100 corpus and query vectors with the authors' (D-005)."""
    questions, index = _dev100(cfg)
    embed = cfg.endpoints["embed"]
    own_dir = cfg.index_dir / f"{embed.model.replace('/', '__')}@{embed.revision}"
    theirs_dir = data_dir / AUTHOR_EMBEDDINGS
    corpus: list[float] = []
    for q in questions:
        own_path = own_dir / f"{q.question_id}.npz"
        if not own_path.exists():
            raise FileNotFoundError(
                f"{own_path}: run the dev100 retrieve stage with own embeddings"
            )
        corpus.extend(
            _row_cosines(
                own_path, theirs_dir / f"buffer/sample{index[q.question_id]}_qwen3emb06b.npz"
            )
        )
    client = Client(Cache(cfg.cache), cfg.concurrency)
    own_q = await client.embed(
        embed, [EMBED_QUERY_PREFIX + q.question for q in questions], op="check"
    )
    with np.load(theirs_dir / "queries/queries_qwen3emb06b.npz", allow_pickle=False) as z:
        rows = {str(qid): i for i, qid in enumerate(z["question_ids"])}
        theirs_q = z["embeddings"][[rows[q.question_id] for q in questions]].astype(np.float64)
    query = _cosines(np.array(own_q, dtype=np.float64), theirs_q)
    return {
        "corpus_mean": float(np.mean(corpus)),
        "corpus_min": float(np.min(corpus)),
        "query_mean": float(np.mean(query)),
        "query_min": float(np.min(query)),
    }


def _row_cosines(own_path: Path, theirs_path: Path) -> list[float]:
    with np.load(own_path, allow_pickle=False) as a, np.load(theirs_path, allow_pickle=False) as b:
        own, theirs = a["embeddings"].astype(np.float64), b["embeddings"].astype(np.float64)
    if own.shape != theirs.shape:
        raise ValueError(f"{own_path}: shape {own.shape} vs authors' {theirs.shape}")
    return [float(x) for x in _cosines(own, theirs)]


def _cosines(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    result: np.ndarray[Any, Any] = num / np.maximum(den, 1e-12)
    return result


def judge_agreement_path(cfg: pipeline.Config) -> Path:
    slug = judge_id(cfg.endpoints["judge"]).replace("/", "__").replace(":", "_")
    return cfg.runs_dir / "judge_agreement" / slug / "verdicts.jsonl"


async def judge_agreement(cfg: pipeline.Config, data_dir: Path) -> tuple[float, int, int]:
    """Our judge on the authors' dev-100 RAG answers: (agreement, judged, unjudged)."""
    questions, _ = _dev100(cfg)
    answers = data.load_author_answers(data_dir / AUTHORS_RAG1_ANSWERS)
    records = await pipeline.judge_external(cfg, questions, answers, judge_agreement_path(cfg))
    return _agreement_with_authors(records, {q.question_id for q in questions}, data_dir)


def _agreement_with_authors(
    records: dict[str, dict[str, Any]], qids: set[str], data_dir: Path
) -> tuple[float, int, int]:
    ours = {
        q: int(r["verdict"]) for q, r in records.items() if q in qids and r["verdict"] is not None
    }
    authors = {
        v.question_id: v.verdict
        for v in data.load_verdicts(data_dir / AUTHORS_RAG1_VERDICTS, "authors")
        if v.question_id in ours
    }
    share = metrics.agreement(ours, authors) if ours else 0.0
    return share, len(ours), len(qids) - len(ours)


def gate1(rag1: pipeline.Config, repeat: pipeline.Config, data_dir: Path) -> list[tuple[str, bool]]:
    t = thresholds()
    checks: list[tuple[str, bool]] = []
    matched, n_hist, _ = chunk_match(data_dir)
    checks.append((f"chunks {matched}/{n_hist}", matched / n_hist >= t["chunk_match_min"]))

    judged = pipeline.read_records(judge_agreement_path(rag1))
    dev_qids = {q.question_id for q in _dev100(rag1)[0]}
    share, n_judged, unjudged = _agreement_with_authors(judged, dev_qids, data_dir)
    checks.append(
        (
            f"judge agreement {share:.3f} on {n_judged} ({unjudged} unjudged)",
            share >= t["judge_agreement_min"] and unjudged == 0,
        )
    )

    questions = pipeline.select_questions(data.load_questions(rag1.questions), rag1.subset)
    verdicts = pipeline.read_records(rag1.run_dir / "verdicts.jsonl")
    parsable = {q: int(r["verdict"]) for q, r in verdicts.items() if r["verdict"] is not None}
    # WHY: a missing or unparsable verdict counts as wrong for accuracy (D-031); the failure
    # check below caps how many there can be.
    accuracy = 100 * sum(parsable.values()) / len(questions)
    within = abs(accuracy - t["rag1_target"]) <= t["rag1_tolerance"]
    checks.append((f"rag1 accuracy {accuracy:.1f} (target {t['rag1_target']})", within))
    authors = {
        v.question_id: v.verdict
        for v in data.load_verdicts(data_dir / AUTHORS_RAG1_VERDICTS, "authors")
        if v.question_id in parsable
    }
    rag_agree = metrics.agreement(parsable, authors) if parsable else 0.0
    checks.append(
        (f"rag1 agreement with authors {rag_agree:.3f}", rag_agree >= t["rag1_agreement_min"])
    )
    failure = (len(questions) - len(parsable)) / len(questions)
    checks.append((f"failed or unparsable {failure:.3%}", failure <= t["failure_rate_max"]))

    leaks = sum(
        bool(r["think_leak"])
        for cfg in (rag1, repeat)
        for r in pipeline.read_records(cfg.run_dir / "answers.jsonl").values()
    )
    checks.append((f"no_think={'true' if leaks == 0 else 'false'} ({leaks} leaks)", leaks == 0))

    rep = pipeline.read_records(repeat.run_dir / "verdicts.jsonl")
    pairs = [(rep[q]["verdict"], verdicts[q]["verdict"]) for q in rep if q in verdicts]
    stable = sum(a == b for a, b in pairs) / len(pairs) if pairs else 0.0
    checks.append(
        (
            f"repeat stability {stable:.3f} on {len(pairs)}",
            stable >= t["repeat_stability_min"] and len(pairs) > 0,
        )
    )
    return checks
