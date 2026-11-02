# Agent rules for memory-tax-replication

Read this first in every session. It is also the project's rulebook for human contributors.

## Session start

Read in this order, and only what you need:

1. `IMPLEMENTATION_PLAN.md`: current phase and gate
2. `ARCHITECTURE.md`: the shape of the system
3. `FLOW.md`: the path you are about to change
4. Newest entries in `DECISION.md` and `BUGS.md`

## Required updates with every change

| When you... | Update | In the same commit? |
| --- | --- | --- |
| Make a meaningful choice (library, pattern, threshold, tradeoff) | `DECISION.md` | Yes |
| Change a call path, or start work on one | `FLOW.md`, including "Currently modifying" | Yes |
| Scope a feature or find a bug | `BUGS.md` | Yes |
| Add or move a module, or change how data moves | `ARCHITECTURE.md` | Yes |
| Make a large or risky edit | `ROLLBACK.md` | Before you start |
| Finish any change | Run `DONE.md` and paste results in the `BUGS.md` entry | Yes |

A decision is meaningful if a reasonable engineer could have chosen otherwise and the choice affects results, cost, or how hard it is to change later.

## Commenting policy

Comment non-obvious logic as you write it. Don't restate the code. Explain the flow: what the block is for, what calls into it, and what it assumes exists.

**Every module** starts with a docstring containing these three labelled lines. `scripts/check_repo.py` fails if one is missing.

```python
"""Replay answers from the authors' SimpleMem stores without any retrieval LLM calls.

Purpose: lets H1 and H2 use all 500 histories, which we could not afford to build.
Called by: pipeline.run_stage("retrieve") when config.system.mode == "replay".
Assumes: stores are extracted under data/stores/ and each id in retrieved_ids exists there.
"""
```

**Inside functions**, use these prefixes wherever the logic isn't obvious:

| Prefix | Use for | Example |
| --- | --- | --- |
| `# WHY:` | A choice the code can't explain | `# WHY: authors store this list as a string; literal_eval never runs code.` |
| `# FLOW:` | What depends on this block | `# FLOW: metrics.py joins on question_id, so it must stay unique per run.` |
| `# ASSUMES:` | A condition the code relies on | `# ASSUMES: verdicts are 0 or 1; judge.py drops unparsable replies first.` |
| `# PAPER:` | Link to its source | `# PAPER: Appendix A.2, the last turn carries over as overlap.` |
| `# shortcut:` | A known limit and when to upgrade | `# shortcut: brute force is enough at ~450 chunks; use an index above ~100k.` |

Don't write comments like `# increment i`. If a comment only restates the line, delete it.

## Engineering rules

- Keep functions small and typed. `mypy src` must pass.
- Add a dependency only with a `DECISION.md` entry.
- Validate data at boundaries: downloaded files, model replies, and config.
- Treat downloaded files as untrusted. Never use `eval` or `pickle` on them. Extract archives with `filter="data"`.
- Read secrets only from environment variables: `HF_TOKEN`, `LLM_API_KEY`, `JUDGE_API_KEY`. Never commit them or write them to logs.
- Make each stage idempotent. A rerun skips finished questions and never overwrites a finished run.
- Every run writes a manifest with its git commit, config hash, model revisions, and dataset revision.
- Non-obvious new logic needs a test. A bug fix needs a test that fails before the fix.

## Research integrity

- `IMPLEMENTATION_PLAN.md` section 3 is preregistered. Don't change a hypothesis, metric, threshold, or test after seeing results without a `DECISION.md` entry and a report note.
- Use dev 100 to tune. Run held-out 400 once, after tag `prereg-h2`. Log any rerun as a deviation.
- Report null and negative results.
- Never say results reproduced unless the gate commands in `DONE.md` passed and their output is recorded.
- Never tune to match the paper's numbers. Log the difference and investigate the cause.

## Stop and ask a human before

- Running the held-out set.
- Changing preregistered items or gate thresholds.
- Adding a paid service or spending money.
- Making anything public on Hugging Face or GitHub, or pushing a release. Hourly checkpoints to your own private dataset are allowed.
- Deleting runs or the cache, force-pushing, or rewriting history.

## Stop conditions

Stop and report when any of these happens:

- The same failure survives 3 fix attempts.
- A step would use more than 2 GPU hours beyond its budget in `IMPLEMENTATION_PLAN.md`.
- A gate fails. Record it in `BUGS.md` first. Don't lower the gate.
