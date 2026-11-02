import asyncio
from typing import Any

from memtax import pipeline
from tests.fake_server import FakeServer


def test_crash_mid_write_and_failed_calls_leave_one_record_each(
    fixture_raw: dict[str, Any], server: FakeServer
) -> None:
    cfg = pipeline.parse_config(fixture_raw)
    path = cfg.run_dir / "retrieval.jsonl"

    # Every request fails: nothing is written, and each question counts as failed.
    server.fail_next = 10_000
    s = asyncio.run(pipeline.run_stage(cfg, "retrieve"))
    assert (s["done"], s["failed"]) == (0, 4)
    server.fail_next = 0

    asyncio.run(pipeline.run_stage(cfg, "retrieve"))
    # Simulate a crash mid-write: cut the last record in half.
    content = path.read_bytes()
    path.write_bytes(content[: len(content) - 40])

    s = asyncio.run(pipeline.run_stage(cfg, "retrieve"))
    assert (s["done"], s["already_done"]) == (1, 3)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 4
    assert len(pipeline.read_records(path)) == 4

    # A finished stage reruns as a no-op.
    s = asyncio.run(pipeline.run_stage(cfg, "retrieve"))
    assert (s["done"], s["already_done"]) == (0, 4)
