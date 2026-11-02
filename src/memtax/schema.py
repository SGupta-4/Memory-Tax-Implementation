"""Shared records passed between stages.

Purpose: one definition of each per-question record, so every stage joins on the same fields.
Called by: data.py builds questions and verdicts; systems/ build Retrieved; answer.py builds
Answer; judge.py builds JudgeResult; pipeline.py writes them as JSONL; metrics.py reads verdicts.
Assumes: nothing outside the standard library; this module imports no other project module.
"""

from dataclasses import dataclass
from typing import NotRequired, TypedDict


class Turn(TypedDict):
    role: str
    content: str
    # PAPER: LongMemEval marks evidence turns with has_answer; most turns omit it.
    has_answer: NotRequired[bool]


@dataclass(frozen=True)
class Question:
    question_id: str
    question_type: str
    question: str
    # WHY: some LongMemEval gold answers are numbers; data.py stores them as text.
    answer: str
    question_date: str
    haystack_session_ids: list[str]
    haystack_dates: list[str]
    haystack_sessions: list[list[Turn]]
    # ASSUMES: empty for abstention questions, which have no gold session.
    answer_session_ids: list[str]


@dataclass(frozen=True)
class Verdict:
    question_id: str
    # ASSUMES: 0 or 1 only; loaders reject anything else, including unparsable judge replies.
    verdict: int
    # FLOW: metrics.pair() refuses to compare verdicts with different judge_id (D-020).
    judge_id: str


@dataclass(frozen=True)
class Retrieved:
    question_id: str
    doc_ids: list[str]
    # FLOW: answer.py joins these, in order, into the shared prompt's memory block.
    texts: list[str]
    round_of: list[int]
    session_ids: list[str]  # source session of each unit, for session-level recall
    # ASSUMES: None for abstention questions, which have no gold session.
    session_recall: float | None
    decisions: list[str]  # raw sufficiency replies (two-round RAG); empty otherwise


@dataclass(frozen=True)
class Answer:
    question_id: str
    answer: str
    prompt_version: str
    think_leak: bool  # "<think>" appeared in the reply; G1 requires none


@dataclass(frozen=True)
class JudgeResult:
    """One judge call as written to verdicts.jsonl; verdict is None when unparsable."""

    question_id: str
    verdict: int | None
    raw: str
    prompt: str
    judge_id: str
