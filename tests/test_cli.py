import json
from pathlib import Path

import pytest

from memtax import cli

ROOT = Path(__file__).parents[1]


def test_smoke_on_fixture(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    assert cli.main(["smoke", "--config", "configs/fixture.yaml"]) == 0
    assert capsys.readouterr().out.strip() == "smoke OK: 4 questions, identity holds"


def test_audit_without_data_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cli.main(["audit", "--bench", "longmemeval_s", "--data-dir", str(tmp_path)])


def _write_audit_layout(root: Path, rag2_correct: int) -> None:
    """Synthetic 500-question layout shaped like the published ledger counts, not real data."""
    ids = [f"q{i:03d}" for i in range(500)]
    question = {
        "question_type": "multi-session",
        "question": "?",
        "answer": "x",
        "question_date": "2023/01/01",
        "haystack_session_ids": [],
        "haystack_dates": [],
        "haystack_sessions": [],
        "answer_session_ids": [],
    }
    (root / "longmemeval").mkdir(parents=True)
    (root / cli.AUDIT_QUESTIONS).write_text(
        json.dumps([{"question_id": q, **question} for q in ids])
    )
    pairs = [(1, 0)] * 41 + [(1, 1)] * 260 + [(0, 1)] * 91 + [(0, 0)] * 108
    verdicts = {
        "simplemem": [p[0] for p in pairs],
        "rag1": [p[1] for p in pairs],
        "rag2": [1] * rag2_correct + [0] * (500 - rag2_correct),
    }
    for name, values in verdicts.items():
        folder = root / cli.AUDIT_RUNS[name]
        folder.mkdir(parents=True)
        lines = [
            json.dumps({"question_id": q, "verdict": v}) for q, v in zip(ids, values, strict=True)
        ]
        (folder / "verdicts.jsonl").write_text("\n".join(lines))


def test_audit_g0_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_audit_layout(tmp_path, rag2_correct=361)
    assert cli.main(["audit", "--bench", "longmemeval_s", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "simplemem=60.2 rag1=70.2 rag2=72.2 tax=18.2 benefit=8.2 n=500"
    assert out[-1] == "G0 PASS"


def test_audit_g0_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_audit_layout(tmp_path, rag2_correct=360)
    assert cli.main(["audit", "--bench", "longmemeval_s", "--data-dir", str(tmp_path)]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "G0 FAIL: rag2=72.0 (want 72.2)"
