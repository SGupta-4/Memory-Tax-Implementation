"""No Context and Gold Context references from Appendix A.2.

Purpose: the floor and oracle references Phase 1 runs next to RAG.
Called by: pipeline.make_system() when config system.name is no_context or gold.
Assumes: has_answer marks the gold evidence turns, as in LongMemEval.
"""

from memtax.retrieval import session_recall
from memtax.schema import Question, Retrieved


class NoContextSystem:
    async def build(self, q: Question) -> None:
        return None

    async def retrieve(self, q: Question) -> Retrieved:
        # FLOW: an empty block becomes the vendored prompt's "(no relevant memory retrieved)".
        return Retrieved(q.question_id, [], [], [], [], None, [])


class GoldContextSystem:
    async def build(self, q: Question) -> None:
        return None

    async def retrieve(self, q: Question) -> Retrieved:
        ids, texts, sessions = [], [], []
        for sid, date, session in zip(
            q.haystack_session_ids, q.haystack_dates, q.haystack_sessions, strict=True
        ):
            for i, turn in enumerate(session):
                if turn.get("has_answer"):
                    # PAPER: the authors render gold evidence as "(<date>) [role: r]: text".
                    ids.append(f"{sid}|t{i}")
                    texts.append(f"({date}) [role: {turn['role']}]: {turn['content']}")
                    sessions.append(sid)
        recall = session_recall(q.answer_session_ids, sessions)
        return Retrieved(q.question_id, ids, texts, [1] * len(ids), sessions, recall, [])
