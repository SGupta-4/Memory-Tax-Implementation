"""Split one LongMemEval history into RAG chunks, following Appendix A.2.

Purpose: produce the raw chunks RAG retrieves over, identical to the authors' `buffer` chunks
so their embeddings can be reused (D-005, D-028).
Called by: systems/rag.py (index and retrieval) and cli check-chunks.
Assumes: sessions, ids and dates are aligned lists, as data.load_questions guarantees.
"""

from dataclasses import dataclass

from memtax.schema import Question

# PAPER: Appendix A.2. Short turns merge into chunks of about 1024 characters; longer turns
# stay alone; turns over 8192 characters split into 8192-character windows, 1024 overlap.
MERGE_CHARS = 1024
WINDOW_CHARS = 8192
WINDOW_OVERLAP = 1024


@dataclass(frozen=True)
class Chunk:
    doc_id: str  # "<question_id>|d<n>", the authors' id form; n counts chunks in history order
    session_id: str
    text: str
    turn_ids: tuple[str, ...]  # "<session_id>|t<turn index>"


def chunk_history(q: Question) -> list[Chunk]:
    raw: list[tuple[str, str, list[str]]] = []  # (session_id, text, turn_ids)
    for sid, date, session in zip(
        q.haystack_session_ids, q.haystack_dates, q.haystack_sessions, strict=True
    ):
        header = f"(session date-time: {date})\n"
        buf: list[tuple[str, str]] = []  # (turn_id, rendered line)

        def emit(items: list[tuple[str, str]], sid: str = sid, header: str = header) -> None:
            raw.append((sid, header + "\n".join(line for _, line in items), [t for t, _ in items]))

        for i, turn in enumerate(session):
            # WHY: the rules below were matched against all 500 authors' histories (D-028):
            # sizes count the rendered "[role: r]: text" lines, without the header.
            content = turn["content"].strip()
            if not content:
                continue
            item = (f"{sid}|t{i}", f"[role: {turn['role']}]: {content}")
            line = item[1]
            if len(line) > WINDOW_CHARS:
                if buf:
                    emit(buf)
                    buf = []
                step = WINDOW_CHARS - WINDOW_OVERLAP
                for start in range(0, len(line), step):
                    raw.append(
                        (sid, header + line[start : start + WINDOW_CHARS].strip(), [item[0]])
                    )
                    if start + WINDOW_CHARS >= len(line):
                        break
            elif len(line) >= MERGE_CHARS:
                if buf:
                    emit(buf)
                    buf = []
                emit([item])
            else:
                if buf and _joined_len(buf + [item]) > MERGE_CHARS:
                    emit(buf)
                    # PAPER: "the last turn carried over as overlap"; the authors carry it only
                    # when the flushed chunk merged more than one turn.
                    buf = [buf[-1]] if len(buf) > 1 else []
                    if buf and _joined_len(buf + [item]) > MERGE_CHARS:
                        emit(buf)
                        buf = []
                buf.append(item)
        if buf:
            emit(buf)
    return [
        Chunk(f"{q.question_id}|d{n}", sid, text, tuple(tids))
        for n, (sid, text, tids) in enumerate(raw)
    ]


def _joined_len(items: list[tuple[str, str]]) -> int:
    return sum(len(line) for _, line in items) + len(items) - 1
