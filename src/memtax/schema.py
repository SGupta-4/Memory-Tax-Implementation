"""Shared records passed between stages.

Purpose: one definition of a question and a verdict, so every stage joins on the same fields.
Called by: data.py builds these records; metrics.py and cli.py consume them.
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
