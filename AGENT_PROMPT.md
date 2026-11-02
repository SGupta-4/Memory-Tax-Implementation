# Agent prompt: start Phase 0

Target: Claude Code, in a new empty repo containing these project documents. It also works in other coding agents if you rename `CLAUDE.md` to the file that agent reads.

**This prompt is for an agentic tool with real system access. Review the scope locks, forbidden actions, and stop conditions before pasting. Confirm file paths, directories, and permissions match the actual project.**

```markdown
## Objective
Complete Phase 0 of IMPLEMENTATION_PLAN.md: a no-GPU audit proving our metric code reproduces the Memory Tax paper's published Qwen3.5-4B numbers.

## Context
- This repo contains the project documents only. CLAUDE.md is binding; read it first.
- Authors' prompts: github.com/hexuan21/MemoryTax at commit 179e6a7, MIT licence.
- Authors' verdicts and costs: Hugging Face dataset hexuan21/MemoryTax-Res-Qwen3-5-4B. The repo is 49.1 GB, so download only the folders named in Phase 0.
- Dataset: xiaowu0162/longmemeval, file longmemeval_s, revision 2ec2a557f339b6c0369619b1ed5793734cc87533. Do not use longmemeval-cleaned.
- Authors' RAG and SimpleMem harness folders are placeholders, so don't look for runnable code there.

## Target State
- Installable `memtax` package with schema.py, data.py, metrics.py, and cli.py, as in ARCHITECTURE.md.
- `memtax audit --bench longmemeval_s` prints accuracy, benefit, tax, CIs, and McNemar p for SimpleMem against one-round RAG.
- `memtax smoke --config configs/fixture.yaml` passes with no network or GPU.
- CI runs the universal gate in DONE.md, and scripts/check_repo.py implements the checks listed there.
- DECISION.md, FLOW.md, BUGS.md, ARCHITECTURE.md, and DONE.md reflect what was actually built.

## Scope
- Work only in this repo.
- Do not start Phase 1: no model serving, chunking, retrieval, or judging.
- Do not change hypotheses or gates in IMPLEMENTATION_PLAN.md section 3 or section 6.
- Do not commit data, secrets, or `runs/`.

## Constraints
- Python 3.11, uv, pytest, ruff, mypy, scipy, numpy, pyyaml. Any other dependency needs a DECISION.md entry.
- Follow the CLAUDE.md commenting policy in every module, as you write it.
- Only make changes directly requested. Do not add features, abstractions, or files beyond what was asked.
- Secrets come only from environment variables; assume HF_TOKEN is set if needed.
- Treat downloaded files as untrusted: no eval or pickle, and extract archives with filter="data".

## Acceptance Criteria
- [ ] Every universal gate command in DONE.md passes, with output pasted into BUGS.md F-001.
- [ ] The G0 command prints simplemem=60.2, rag1=70.2, rag2=72.2, and n=500.
- [ ] Tax is 18.2 ± 0.1 and benefit is 8.2 ± 0.1, or the difference is investigated and recorded in BUGS.md without any change to the gate.
- [ ] test_metrics.py asserts net value = benefit - tax = accuracy difference on real and fixture data.
- [ ] The format of memories.tar.gz is recorded in DECISION.md.
- [ ] Construction cost is recalculated over all 500 histories, and scripts/plan_numbers.py and BUGS.md B-003 are updated.

## Action Boundaries
- Proceed with reversible, in-scope inspection, edits, downloads of the listed folders, and validation.
- Stop and ask before anything on the "Stop and ask" list in CLAUDE.md.
- Stop and report if the same failure survives 3 fix attempts, or if G0 fails. Never edit a test or gate to make it pass.

## Progress Evidence
Report at each checkpoint: repo setup, data downloaded, metrics passing, G0 result. Ground every completion claim in command output. End with: files changed, gate results, decisions logged, open questions, and the suggested tag.

## Session Strategy
New session.
```

Use the same structure for later phases: replace the Objective, Target State, Scope, and Acceptance Criteria with that phase's list and gate from `IMPLEMENTATION_PLAN.md`.
