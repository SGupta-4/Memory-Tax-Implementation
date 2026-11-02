import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from memtax import pipeline
from memtax.retrieval import BM25, session_recall, top_k, union
from memtax.systems.rag import parse_sufficiency
from tests.fake_server import FakeServer


def _run_all(raw: dict[str, Any]) -> pipeline.Config:
    cfg = pipeline.parse_config(raw)
    for stage in pipeline.STAGE_FILES:
        s = asyncio.run(pipeline.run_stage(cfg, stage))
        assert s["failed"] == s["missing_input"] == 0
    return cfg


def _records(cfg: pipeline.Config, stage: str) -> dict[str, dict[str, Any]]:
    return pipeline.read_records(cfg.run_dir / pipeline.STAGE_FILES[stage])


def test_full_run_writes_manifest_and_records(fixture_raw: dict[str, Any]) -> None:
    cfg = _run_all(fixture_raw)
    manifest = json.loads((cfg.run_dir / "manifest.json").read_text())
    assert manifest["dataset"]["sha256"] == fixture_raw["questions_sha256"]
    assert manifest["judge_id"] == "fake-judge@fake:judge/unified-v1"
    assert [r["stage"] for r in manifest["stage_runs"]] == ["retrieve", "answer", "judge"]
    retrieved = _records(cfg, "retrieve")
    assert all(len(r["doc_ids"]) <= 2 for r in retrieved.values())  # rerank_k = 2
    assert retrieved["fx_ssu_004_abs"]["session_recall"] is None
    assert all(r["verdict"] in (0, 1) for r in _records(cfg, "judge").values())


def test_two_rounds_add_round_two_units(fixture_raw: dict[str, Any]) -> None:
    # WHY: fx_ssu_001 has 2 chunks; k=1 leaves one for round two to find.
    fixture_raw["system"].update(rounds=2, bm25_k=1, dense_k=1, rerank_k=1)
    cfg = _run_all(fixture_raw)
    r = _records(cfg, "retrieve")["fx_ssu_001"]
    assert r["decisions"] and 2 in r["round_of"]  # fake server always says "not sufficient"
    assert len(set(r["doc_ids"])) == len(r["doc_ids"])  # round two searches the remaining pool


def test_run_id_ignores_paths_and_concurrency(fixture_raw: dict[str, Any]) -> None:
    a = pipeline.parse_config(fixture_raw)
    b = pipeline.parse_config({**fixture_raw, "concurrency": 1, "runs_dir": "elsewhere"})
    c = pipeline.parse_config({**fixture_raw, "subset": "dev100"})
    assert a.run_id == b.run_id != c.run_id


def test_think_leak_is_flagged(fixture_raw: dict[str, Any], server: FakeServer) -> None:
    server.think = True
    cfg = _run_all(fixture_raw)
    assert all(r["think_leak"] for r in _records(cfg, "answer").values())


def test_baselines(fixture_raw: dict[str, Any]) -> None:
    fixture_raw["system"] = {"name": "gold"}
    gold = _records(_run_all(fixture_raw), "retrieve")
    assert gold["fx_ms_002"]["doc_ids"] == ["s3|t0", "s4|t0"]
    assert gold["fx_ms_002"]["session_recall"] == 1.0
    fixture_raw["system"] = {"name": "no_context"}
    assert _records(_run_all(fixture_raw), "retrieve")["fx_ms_002"]["texts"] == []


def test_dev_and_heldout_partition_500() -> None:
    from memtax import vendor

    ids = [q["question_id"] for q in json.loads(vendor.DEV_SUBSET.read_text())["questions"]]
    assert len(ids) == len(set(ids)) == 100


def test_shards_partition_questions(fixture_raw: dict[str, Any]) -> None:
    from memtax import data

    qs = data.load_questions(Path(fixture_raw["questions"]))
    parts = [pipeline.shard(qs, i, 3) for i in range(3)]
    assert sorted(q.question_id for p in parts for q in p) == sorted(q.question_id for q in qs)
    with pytest.raises(ValueError):
        pipeline.shard(qs, 3, 3)


@pytest.mark.parametrize(
    "bad",
    [{"subset": "test"}, {"system": {"name": "x"}}, {"questions_sha256": None}],
)
def test_bad_config_is_rejected(fixture_raw: dict[str, Any], bad: dict[str, Any]) -> None:
    raw = {**fixture_raw, **bad}
    if bad.get("questions_sha256", "") is None:
        del raw["questions_sha256"]
    with pytest.raises(ValueError, match="config"):
        pipeline.parse_config(raw)


def test_missing_revision_is_rejected(fixture_raw: dict[str, Any]) -> None:
    del fixture_raw["endpoints"]["judge"]["revision"]
    with pytest.raises(ValueError, match="revision"):
        pipeline.parse_config(fixture_raw)


def test_bm25_prefers_matching_rare_terms() -> None:
    bm25 = BM25(["the cat sat", "the dog sat", "the the the"])
    scores = bm25.scores("cat")
    assert top_k(scores, 1, [0, 1, 2]) == [0]
    assert top_k(scores, 2, [1, 2]) == [1, 2]  # restricted pool, ties keep chunk order


def test_union_and_recall() -> None:
    assert union([3, 1], [1, 2]) == [3, 1, 2]
    assert session_recall(["a", "b"], ["a", "a", "c"]) == 0.5
    assert session_recall([], ["a"]) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"sufficient": true, "next_query": ""}', (True, "Q")),
        ('Sure: {"sufficient": false, "next_query": "when?"}', (False, "when?")),
        ('{"sufficient": false, "next_query": ""}', (False, "Q")),
        ("no json here", (True, "Q")),
        ('{"sufficient": "yes"}', (True, "Q")),
    ],
)
def test_parse_sufficiency(raw: str, expected: tuple[bool, str]) -> None:
    assert parse_sufficiency(raw, "Q") == expected
