import json
from pathlib import Path

import pytest

from memtax import data

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_fixture_questions() -> None:
    qs = data.load_questions(FIXTURES / "longmemeval_fixture.json")
    assert len(qs) == 4
    assert qs[1].answer == "2"  # numeric gold answer stored as text
    assert qs[3].answer_session_ids == []  # abstention question
    assert qs[0].haystack_sessions[0][0].get("has_answer") is True


def test_question_missing_field_is_rejected(tmp_path: Path) -> None:
    item = json.loads((FIXTURES / "longmemeval_fixture.json").read_text())[0]
    del item["question_date"]
    path = tmp_path / "q.json"
    path.write_text(json.dumps([item]))
    with pytest.raises(ValueError, match="question_date"):
        data.load_questions(path)


def test_verdicts_found_in_nested_files(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.json").write_text(
        json.dumps({"results": [{"question_id": "b", "label": "CORRECT"}]})
    )
    (tmp_path / "b.jsonl").write_text('{"question_id": "a", "verdict": false}\n')
    # A QA-output record without a verdict is skipped, not an error.
    (tmp_path / "qa.json").write_text(json.dumps([{"question_id": "a", "answer": "x"}]))
    vs = data.load_verdicts(tmp_path, "j")
    assert [(v.question_id, v.verdict) for v in vs] == [("a", 0), ("b", 1)]


def test_unparsable_verdict_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "v.jsonl").write_text('{"question_id": "a", "verdict": null}\n')
    with pytest.raises(ValueError, match="not 0/1"):
        data.load_verdicts(tmp_path, "j")


def test_duplicate_verdict_is_rejected(tmp_path: Path) -> None:
    line = '{"question_id": "a", "verdict": 1}\n'
    (tmp_path / "v.jsonl").write_text(line * 2)
    with pytest.raises(ValueError, match="second verdict"):
        data.load_verdicts(tmp_path, "j")


def test_no_verdicts_reports_keys_seen(tmp_path: Path) -> None:
    (tmp_path / "v.json").write_text(json.dumps([{"question_id": "a", "judge_raw": "?"}]))
    with pytest.raises(ValueError, match="judge_raw"):
        data.load_verdicts(tmp_path, "j")
