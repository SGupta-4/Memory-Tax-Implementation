"""Score an answer with the authors' judge prompts and verdict parser.

Purpose: one judge path for every system; the judge never sees which system answered.
Called by: pipeline.py, judge stage.
Assumes: the vendored judge prompts at 179e6a7; temperature 0 and reasoning off come from llm.py.
"""

from memtax import vendor
from memtax.llm import Client, Endpoint
from memtax.schema import Answer, JudgeResult, Question

J = vendor.load("common/judge/prompts.py")
BENCH = "longmemeval_s"


def judge_id(ep: Endpoint) -> str:
    # FLOW: metrics.pair() refuses to compare verdicts whose judge_id differs (D-020).
    return f"{ep.model}@{ep.revision}:{J.JUDGE_PROMPT_VERSION}"


async def judge(client: Client, ep: Endpoint, q: Question, ans: Answer) -> JudgeResult:
    name = str(J.pick_prompt(BENCH, q.question_type))
    messages = J.build_messages(BENCH, q.question_type, q.question, [q.answer], ans.answer)
    raw, verdict = "", None
    # PAPER: FLOW.md, an unparsable verdict is retried once, then flagged as None.
    for attempt in (0, 1):
        raw = await client.chat(
            ep,
            messages,
            op="judge",
            prompt_version=str(J.JUDGE_PROMPT_VERSION),
            max_tokens=int(J.MAX_TOKENS[name]),
            attempt=attempt,
        )
        verdict = J.parse_verdict(raw)
        if verdict is not None:
            break
    return JudgeResult(q.question_id, verdict, raw, name, judge_id(ep))
