"""Load and validate LongMemEval questions and judge verdicts from downloaded files.

Purpose: the trust boundary for downloaded data, so later code sees only validated records.
Called by: cli.audit() and cli.smoke(); later phases add the Benchmark protocol here.
Assumes: files were downloaded as in FLOW.md "Phase 0 data layout"; contents are untrusted
JSON, so this module parses with json only and never uses eval or pickle.
"""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from memtax.schema import Question, Turn, Verdict

_QUESTION_LIST_FIELDS = (
    "haystack_session_ids",
    "haystack_dates",
    "haystack_sessions",
    "answer_session_ids",
)

# WHY: the authors' verdict layout could not be inspected (HF blocked, B-004), so the loader
# accepts these field names in priority order and fails loudly if none is present.
VERDICT_KEYS = ("verdict", "judge_verdict", "label", "judge_label", "correct", "is_correct")

# PAPER: B-001; Hub LFS sha256 of `longmemeval_s` at revision 2ec2a557f3, verified 2026-11-02.
LONGMEMEVAL_S_SHA256 = "08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894"


def verify_sha256(path: Path, expected: str) -> None:
    with path.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{path}: sha256 {actual} is not the pinned {expected}")


def load_questions(path: Path) -> list[Question]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of questions")
    questions = [_parse_question(item, path) for item in raw]
    ids = [q.question_id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate question_id")
    return questions


def _parse_question(item: Any, path: Path) -> Question:
    if not isinstance(item, dict):
        raise ValueError(f"{path}: question is not an object")
    qid = item.get("question_id")
    for key in ("question_id", "question_type", "question", "question_date"):
        if not isinstance(item.get(key), str):
            raise ValueError(f"{path}: {qid}: field {key!r} missing or not text")
    if not isinstance(item.get("answer"), str | int | float):
        raise ValueError(f"{path}: {qid}: field 'answer' missing or not text or number")
    for key in _QUESTION_LIST_FIELDS:
        if not isinstance(item.get(key), list):
            raise ValueError(f"{path}: {qid}: field {key!r} missing or not a list")
    sessions = [_parse_session(s, path, qid) for s in item["haystack_sessions"]]
    if not len(sessions) == len(item["haystack_session_ids"]) == len(item["haystack_dates"]):
        raise ValueError(f"{path}: {qid}: haystack sessions, ids and dates differ in length")
    return Question(
        question_id=item["question_id"],
        question_type=item["question_type"],
        question=item["question"],
        answer=str(item["answer"]),
        question_date=item["question_date"],
        haystack_session_ids=[str(s) for s in item["haystack_session_ids"]],
        haystack_dates=[str(d) for d in item["haystack_dates"]],
        haystack_sessions=sessions,
        answer_session_ids=[str(s) for s in item["answer_session_ids"]],
    )


def _parse_session(session: Any, path: Path, qid: Any) -> list[Turn]:
    if not isinstance(session, list):
        raise ValueError(f"{path}: {qid}: a haystack session is not a list")
    turns: list[Turn] = []
    for t in session:
        if not (
            isinstance(t, dict)
            and isinstance(t.get("role"), str)
            and isinstance(t.get("content"), str)
        ):
            raise ValueError(f"{path}: {qid}: a turn lacks text role or content")
        turn: Turn = {"role": t["role"], "content": t["content"]}
        if "has_answer" in t:
            turn["has_answer"] = bool(t["has_answer"])
        turns.append(turn)
    return turns


def load_verdicts(path: Path, judge_id: str) -> list[Verdict]:
    """Read every verdict under a file or folder; exactly one per question_id."""
    found: dict[str, Verdict] = {}
    keys_seen: set[str] = set()
    for file in _json_files(path):
        for record in _records(file):
            keys_seen.update(record)
            key = next((k for k in VERDICT_KEYS if k in record), None)
            if key is None:
                # WHY: QA-output records also carry question_id but no verdict; skip them.
                continue
            qid = str(record["question_id"])
            if qid in found:
                raise ValueError(f"{file}: second verdict for question {qid}")
            found[qid] = Verdict(qid, _parse_verdict(record[key], file, qid), judge_id)
    if not found:
        raise ValueError(
            f"{path}: no verdicts found; expected one of {VERDICT_KEYS} next to question_id. "
            f"Keys seen: {sorted(keys_seen)[:30]}"
        )
    return [found[qid] for qid in sorted(found)]


def _json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"{path}: not found; see FLOW.md for the download commands")
    return sorted(p for p in path.rglob("*") if p.suffix in (".json", ".jsonl") and p.is_file())


def _records(file: Path) -> Iterator[dict[str, Any]]:
    text = file.read_text(encoding="utf-8")
    if file.suffix == ".jsonl":
        docs = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        docs = [json.loads(text)]
    for doc in docs:
        yield from _walk(doc)


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    # WHY: result files may wrap records in lists or keyed objects; any object with a
    # question_id is a record, and its children are not searched further.
    if isinstance(node, dict):
        if "question_id" in node:
            yield node
            return
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _parse_verdict(value: Any, file: Path, qid: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip().upper() in ("CORRECT", "WRONG"):
        return 1 if value.strip().upper() == "CORRECT" else 0
    # WHY: an unparsable verdict needs a recorded policy (DECISION.md), not a silent default.
    raise ValueError(f"{file}: question {qid}: verdict {value!r} is not 0/1 or CORRECT/WRONG")
