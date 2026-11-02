"""Answer a question from retrieved memory with the authors' shared QA prompt.

Purpose: every system answers through the same prompt, so differences come from memory.
Called by: pipeline.py, answer stage.
Assumes: Retrieved.texts are already rendered units; the vendored prompt is at 179e6a7.
"""

from memtax import vendor
from memtax.llm import Client, Endpoint
from memtax.schema import Answer, Question, Retrieved

QA = vendor.load("common/qa_prompts/qa_longmemeval.py")
PROMPT_VERSION = str(QA.PROMPT_VERSION)


async def answer(client: Client, ep: Endpoint, q: Question, retrieved: Retrieved) -> Answer:
    # WHY: the authors' RAG rendering isn't published; units are joined by blank lines (D-030).
    messages = QA.build_messages(
        memory_block="\n\n".join(retrieved.texts),
        question_date=q.question_date,
        question=q.question,
        question_type=q.question_type,
    )
    text = await client.chat(
        ep,
        messages,
        op="qa_answer",
        prompt_version=PROMPT_VERSION,
        max_tokens=int(QA.ANSWER_MAX_TOKENS),
    )
    return Answer(q.question_id, text, PROMPT_VERSION, "<think>" in text)
