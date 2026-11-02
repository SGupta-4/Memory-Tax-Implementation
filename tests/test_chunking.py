from memtax.chunking import chunk_history
from memtax.schema import Question, Turn

HEADER = "(session date-time: D)\n"


def _q(*sessions: list[Turn]) -> Question:
    return Question(
        question_id="q",
        question_type="multi-session",
        question="?",
        answer="a",
        question_date="D",
        haystack_session_ids=[f"s{i}" for i in range(len(sessions))],
        haystack_dates=["D"] * len(sessions),
        haystack_sessions=list(sessions),
        answer_session_ids=[],
    )


def _t(n: int, role: str = "user") -> Turn:
    return {"role": role, "content": "x" * n}


def _turns(q: Question) -> list[tuple[str, ...]]:
    return [c.turn_ids for c in chunk_history(q)]


def test_short_turns_merge_and_never_cross_sessions() -> None:
    chunks = chunk_history(_q([_t(10), _t(10, "assistant")], [_t(10)]))
    assert [c.turn_ids for c in chunks] == [("s0|t0", "s0|t1"), ("s1|t0",)]
    assert (
        chunks[0].text == HEADER + "[role: user]: " + "x" * 10 + "\n[role: assistant]: " + "x" * 10
    )
    assert [c.doc_id for c in chunks] == ["q|d0", "q|d1"]


def test_overlap_carries_last_turn_only_after_a_merge() -> None:
    # Lines are 414 chars ("[role: user]: " + 400): two fit in 1024, three do not.
    q = _q([_t(400), _t(400), _t(400), _t(900)])
    assert _turns(q) == [
        ("s0|t0", "s0|t1"),
        ("s0|t1", "s0|t2"),  # t1 carried over
        ("s0|t2",),  # carried t2 can't join the 914-char line, so it is flushed alone
        ("s0|t3",),
    ]


def test_no_overlap_after_a_single_turn_chunk() -> None:
    assert _turns(_q([_t(700), _t(700)])) == [("s0|t0",), ("s0|t1",)]


def test_line_of_1024_or_more_stands_alone() -> None:
    q = _q([_t(5), _t(1024 - len("[role: user]: ")), _t(5)])
    assert _turns(q) == [("s0|t0",), ("s0|t1",), ("s0|t2",)]


def test_long_turn_splits_into_overlapping_windows() -> None:
    chunks = chunk_history(_q([_t(16000)]))
    line_len = 16000 + len("[role: user]: ")
    bodies = [c.text.removeprefix(HEADER) for c in chunks]
    assert [len(b) for b in bodies] == [8192, 8192, line_len - 14336]
    assert all(c.turn_ids == ("s0|t0",) for c in chunks)


def test_empty_turns_are_skipped_and_text_is_stripped() -> None:
    q = _q([{"role": "user", "content": "  hi  "}, {"role": "assistant", "content": " "}])
    chunks = chunk_history(q)
    assert [(c.text, c.turn_ids) for c in chunks] == [(HEADER + "[role: user]: hi", ("s0|t0",))]
