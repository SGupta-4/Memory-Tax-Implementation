# Plan review

**Verdict:** approved to start Phase 0. The plan's process meets an industry standard, but the project earns that label only after G5 passes.

**Who reviewed it:** I reviewed my own plan in two roles, without an independent reviewer. Verdicts are judgement calls. Treat them as a structured self-check, and ask a mentor or peer for a second opinion before Phase 3.

## Iteration history

| Iteration | What was reviewed | Result |
| --- | --- | --- |
| 0 | Earlier 3-week plan, checked against the authors' files | Failed: wrong dataset, underpowered, and build budget far too low |
| 1 | First draft of this package | Failed: 13 of the findings below |
| 2 | Draft with those fixes, plus a cross-document check | Failed: finding 6, finding 8, a rounding mismatch, and a dead link |
| 3 | Final package | Passed, with the conditions below |

## Findings and fixes

### Project manager

| # | Finding | Severity | Fix |
| --- | --- | --- | --- |
| 1 | Phase 1 packed the whole harness and first Kaggle use into 5 days | High | Fixture and fake server first; 7 days; total now about 30 days |
| 2 | No view of quota by phase | Medium | Added hours per phase and a 35-hour trigger |
| 3 | Agent rule banned HF publishing, but runs need private checkpoints | High | Private checkpoints allowed; public release needs approval |
| 4 | CC-BY results and other licences had no attribution task | Medium | Added `THIRD_PARTY.md` to Phase 5 |
| 5 | No repo layout for the agent to follow | Medium | Added to `ARCHITECTURE.md` |
| 6 | The 12-hour build fallback contradicted the 2-hour overrun stop rule | High | Fallback is now 10 hours, or 8.3 + about 2 |
| 7 | Commands used in `DONE.md` weren't listed in `FLOW.md` | Low | Added them; the check now covers it |
| 8 | No regular checkpoint | Low | Added a review at the end of every phase |

### AI researcher

| # | Finding | Severity | Fix |
| --- | --- | --- | --- |
| 9 | Comparing your judge's verdicts with the authors' could create a fake tax | High | `judge_id` is stored, and mixed pairs are refused ([D-020](DECISION.md)) |
| 10 | Cache could hide non-deterministic answers | Medium | Uncached repeat of 20 dev questions, reported in G1 |
| 11 | Paper doesn't specify source-augmentation pools or budget | High | Made explicit in [D-019](DECISION.md); H2 is framed as testing this version |
| 12 | Paper's appendix doesn't include the two-round sufficiency prompt | Medium | Two-round RAG is no longer gated |
| 13 | Retrieval Scope appeared in prerequisites, but no task calculated it | Medium | Added to Phase 2 |
| 14 | Ten builds could be mistaken for proof of fidelity | Medium | Plan and report must state the limit |
| 15 | Held-out rule didn't address baseline verdicts seen earlier | Low | Rule now says baselines may be seen, and source augmentation may not |

## Final scorecard

| Criterion | PM verdict | Evidence |
| --- | --- | --- |
| Scope, with a defined stopping point | Pass | Minimum shippable after Phase 3; scope ladder |
| Budget from measurements | Pass, with one condition | Authors' cost files; T4 speed still assumed |
| Risks with early signals | Pass | Plan section 8 |
| Binary gates | Pass | `DONE.md` |
| Rollback path | Pass | `ROLLBACK.md` |
| Works if an agent or person changes | Pass | Control documents and `CLAUDE.md` |

| Criterion | Researcher verdict | Evidence |
| --- | --- | --- |
| Clear, falsifiable hypotheses | Pass | Plan section 3 |
| Adequate power for confirmatory claims | Pass for H1 and for H2 at about 6 points | `scripts/plan_numbers.py` |
| Fair comparison | Pass | Same prompts, same judge, paired tests |
| Fidelity shown before new claims | Pass as a design | G0, G1, and G2 |
| Protection from overfitting | Pass | Dev and held-out split, preregistration tag |
| New contribution | Pass, as a pilot | H2 on 4B; medication benchmark |
| Honest about limits | Pass | Reused stores, exploratory H3–H5 |

## Conditions on approval

These can't be settled by planning. Each has a gate.

1. **G0:** Your audit must reproduce the authors' numbers. The expected tax was read from a figure, not calculated from verdict files.
2. **Phase 0:** `memories.tar.gz` must load. I only confirmed its size and existence.
3. **Phase 1:** Qwen3.5-4B must run on Kaggle's T4s, and T4 speed must be measured. The 3× slowdown is an assumption.
4. **Phase 1:** A judge must reach 90% agreement. The 27B GGUF hasn't been tested on T4s.
5. **Phase 1:** Kaggle's actual quota must be checked in your account.

## What was deliberately left out

Docker, MLflow, Airflow, a vector database, and more memory systems. [D-011](DECISION.md) and [D-012](DECISION.md) explain why, and when to revisit that. They would look more "enterprise", but wouldn't make these results more trustworthy.
