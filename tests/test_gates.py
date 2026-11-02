import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from memtax import chunking, data, gates, pipeline, vendor
from memtax.llm import EMBED_QUERY_PREFIX
from tests.fake_server import _embed


@pytest.fixture
def authors_dir(
    fixture_raw: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A data dir shaped like the authors' downloads, built from the 4-question fixture."""
    d = tmp_path / "data"
    qs = data.load_questions(Path(fixture_raw["questions"]))
    (d / "longmemeval").mkdir(parents=True)
    shutil.copy(fixture_raw["questions"], d / gates.QUESTIONS)
    chunks = [
        [
            {
                "doc_id": c.doc_id,
                "text": c.text,
                "question_id": q.question_id,
                "source_turn_ids": list(c.turn_ids),
            }
            for c in chunking.chunk_history(q)
        ]
        for q in qs
    ]
    (d / gates.AUTHOR_CHUNKS).parent.mkdir(parents=True)
    (d / gates.AUTHOR_CHUNKS).write_text(json.dumps(chunks))
    (d / gates.AUTHORS_RAG1_ANSWERS).parent.mkdir(parents=True)
    (d / gates.AUTHORS_RAG1_ANSWERS).write_text(
        "\n".join(json.dumps({"question_id": q.question_id, "pred": q.answer}) for q in qs)
    )
    (d / gates.AUTHORS_RAG1_VERDICTS).parent.mkdir(parents=True)
    (d / gates.AUTHORS_RAG1_VERDICTS).write_text(
        "\n".join(json.dumps({"question_id": q.question_id, "verdict": 1}) for q in qs)
    )
    # WHY: the fixture ids aren't in the authors' dev 100, so make them the dev set here.
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"questions": [{"question_id": q.question_id} for q in qs]}))
    monkeypatch.setattr(vendor, "DEV_SUBSET", dev)
    return d


def test_gate1_reads_every_criterion(fixture_raw: dict[str, Any], authors_dir: Path) -> None:
    rag1 = pipeline.parse_config(fixture_raw)
    repeat = pipeline.parse_config({**fixture_raw, "limit": 2, "use_cache": False})
    for cfg in (rag1, repeat):
        for stage in pipeline.STAGE_FILES:
            asyncio.run(pipeline.run_stage(cfg, stage))
    share, judged, unjudged = asyncio.run(gates.judge_agreement(rag1, authors_dir))
    # The fake judge marks the authors' answers (the gold text) CORRECT, as do their verdicts.
    assert (share, judged, unjudged) == (1.0, 4, 0)

    checks = dict(gates.gate1(rag1, repeat, authors_dir))
    names = [k.split(" ")[0] for k in checks]
    assert names == ["chunks", "judge", "rag1", "rag1", "failed", "no_think=true", "repeat"]
    assert checks["chunks 4/4"] and checks["judge agreement 1.000 on 4 (0 unjudged)"]
    assert checks["failed or unparsable 0.000%"]
    assert checks["repeat stability 1.000 on 2"]


def test_check_embeddings_against_matching_authors_files(
    fixture_raw: dict[str, Any], authors_dir: Path
) -> None:
    cfg = pipeline.parse_config(fixture_raw)
    asyncio.run(pipeline.run_stage(cfg, "retrieve"))  # builds own indexes
    qs = data.load_questions(cfg.questions)
    embed = cfg.endpoints["embed"]
    own = cfg.index_dir / f"{embed.model}@{embed.revision}"
    theirs = authors_dir / gates.AUTHOR_EMBEDDINGS
    (theirs / "buffer").mkdir(parents=True)
    (theirs / "queries").mkdir()
    for i, q in enumerate(qs):
        shutil.copy(own / f"{q.question_id}.npz", theirs / f"buffer/sample{i}_qwen3emb06b.npz")
    np.savez(
        theirs / "queries/queries_qwen3emb06b.npz",
        question_ids=np.array([q.question_id for q in qs]),
        embeddings=np.array([_embed(EMBED_QUERY_PREFIX + q.question) for q in qs]),
    )
    r = asyncio.run(gates.check_embeddings(cfg, authors_dir))
    assert r["corpus_min"] == pytest.approx(1.0) and r["query_min"] == pytest.approx(1.0)
