# Memory Tax: Full Implementation Plan

**Status:** ready for Phase 0 · **Date:** 2026-11-02 · **Replaces:** the earlier 3-week plan doc

## 1. Summary

Replicate "Memory Tax" (He et al., 2026, [arXiv 2610.34073](https://arxiv.org/abs/2610.34073)) on all 500 LongMemEval-s questions, test its fix on a model the paper's fix table doesn't cover, and add a new medication-history benchmark.

The plan runs free on Kaggle. It needs about 22 GPU hours and about 30 focused days.

The key design choice is to reuse what the authors released and spend GPU time only where it creates new evidence.

| Question | Answer |
| --- | --- |
| What is being tested? | Whether SimpleMem loses to plain RAG on Qwen3.5-4B, and whether retrieving raw history alongside memory reduces that loss |
| What is new? | Source augmentation on Qwen3.5-4B, and a medication-history benchmark |
| What is reused? | Authors' prompts, verdicts, chunks, embeddings, and SimpleMem stores |
| What do you build? | RAG, the evaluation harness, metrics, source augmentation, analysis, and the benchmark |
| What does it cost? | Nothing, if the free Kaggle quota holds |

The authors' [repo](https://github.com/hexuan21/MemoryTax) has benchmark and prompt code, but its RAG and SimpleMem harness folders are still placeholders. That leaves real implementation work for you.

## 2. Why this differs from the earlier plan

Checking the authors' actual files found three problems. [BUGS.md](BUGS.md) traces each one.

| Earlier plan | Problem found | This plan |
| --- | --- | --- |
| Use `longmemeval-cleaned` | The authors used the original dataset, so per-question comparison would not be valid | Original `longmemeval_s` at revision `2ec2a557f3` |
| Run 100 questions | Only 42% power to detect the published 10-point gap; that subset was for GPT-5.6 Luna | All 500, with 100 for development and 400 held out |
| Build SimpleMem on 100 histories | About 250,000 generated tokens per history, or 27.6 hours even on the authors' GPUs | Build 10 to check fidelity; reuse the authors' 500 stores |

## 3. Hypotheses and statistics

Lock these before any full run. Changing them later needs a [DECISION.md](DECISION.md) entry and a note in the report.

| ID | Hypothesis | Test set | Status |
| --- | --- | --- | --- |
| H1 | Your one-round RAG beats SimpleMem replayed from the authors' stores | 500 | Confirmatory |
| H2 | SimpleMem plus source augmentation beats SimpleMem | Held-out 400 | Confirmatory, new evidence |
| H3 | SimpleMem's tax is highest on exact-dose and passing-allergy questions | 360 medication | Exploratory pilot |
| H4 | SimpleMem shows a benefit on time-order questions | 360 medication | Exploratory pilot |
| H5 | Source augmentation reduces tax on exact-dose questions | 360 medication | Exploratory pilot |

**Metrics**

```text
accuracy  = mean(verdict)
benefit   = P(system correct and reference wrong)
tax       = P(system wrong and reference correct)
net value = benefit - tax = accuracy_system - accuracy_reference
```

The reference is always one-round RAG. The last line is an identity, so a test asserts it exactly.

**Statistical rules**

- Use exact McNemar tests on paired verdicts.
- Report 95% bootstrap CIs with 10,000 resamples and a fixed seed.
- Apply Holm correction across H1 and H2.
- Treat per-type results as exploratory.
- On medication data, resample by patient, because questions from one history are not independent.
- Report null results the same way as positive ones.
- Compare only runs scored by the same judge and judge version. `metrics.py` refuses mixed pairs, and audits of the authors' verdicts stay separate.
- Earlier phases may see baseline verdicts on the held-out questions. No source-augmentation run or tuning can touch those questions before tag `prereg-h2`.

**Power, from `scripts/plan_numbers.py`**

| Effect | n=100 | n=400 | n=500 |
| --- | --- | --- | --- |
| H1: tax 18.2%, benefit 8.2% | 0.42 | 0.97 | 0.99 |
| H2: +6 points (assumed 9% wins, 3% losses) | 0.31 | 0.93 | 0.97 |
| H2: +3 points (assumed 6% wins, 3% losses) | 0.09 | 0.45 | 0.56 |

H2 can reliably detect a gain near 6 points, not 3. The paper's gain on Qwen3.6-27B was 9.2 points, but a smaller model may behave differently. If H2 is null, report it as "no gain above roughly 6 points detected".

## 4. Pinned inputs

| Item | Value |
| --- | --- |
| Dataset | [`xiaowu0162/longmemeval`](https://huggingface.co/datasets/xiaowu0162/longmemeval), file `longmemeval_s`, revision `2ec2a557f339b6c0369619b1ed5793734cc87533`, 278 MB |
| Evidence labels | Turn level `has_answer`; session level `answer_session_ids` |
| Authors' code | `hexuan21/MemoryTax` at `179e6a7`, MIT |
| Dev subset | `common/bench/subset_ids/longmemeval_s_sub100.json` |
| Authors' results | [`MemoryTax-Res-Qwen3-5-4B`](https://huggingface.co/datasets/hexuan21/MemoryTax-Res-Qwen3-5-4B); revision `987bfe5fb8317900abbad3a3260672299db1229c`; download only the folders in Phase 0, as the repo is 49.1 GB |
| Oracle results and chunks | [`MemoryTax-Res-Other`](https://huggingface.co/datasets/hexuan21/MemoryTax-Res-Other): `res_with_oracle_mem`, `preprocess_chunks`, `preprocess_embeddings` |
| SimpleMem | [`aiming-lab/SimpleMem`](https://github.com/aiming-lab/SimpleMem) at `db80b6a7c591e0ea730a058e9f5fc4eb06572299`, not `main` |
| SimpleMem settings | Window 10; retrieval k: semantic 25, keyword 5, structured 5 |
| Answer model | `Qwen/Qwen3.5-4B`, temperature 0, reasoning off |
| Retrieval models | `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Reranker-0.6B` |
| Judge | `Qwen/Qwen3.6-27B`, temperature 0, reasoning off; [fallback in D-009](DECISION.md) |
| RAG settings | Buffer chunks of about 1,024 characters; BM25 and dense top 10 each; rerank to top 10 |

## 5. Efficiency and scale

| Technique | What it saves or enables |
| --- | --- |
| Reuse authors' stores | Avoids about 138 reference-GPU hours of building all 500 |
| Verify, then reuse chunks and embeddings | Avoids 1.7 GPU hours and tests your chunker exactly |
| Reuse oracle results | Removes the LLM mapping of gold turns to memory units |
| SQLite call cache keyed on model, revision, prompt, and settings | A dead session or rerun never pays twice |
| Stages that write per-question records and skip finished ones | Sessions can stop at any point and resume |
| Separate answer and judge sessions | The 27B judge and the answer stack never compete for GPU memory |
| Sharding with `--shard i/n` | Large runs split across sessions without code changes |
| Brute-force NumPy search per history | About 450 chunks per history needs no vector database |
| Bounded async clients | Sends parallel requests up to a set limit, without overloading the server |
| One `MemorySystem` interface and one `Benchmark` interface | A new system or benchmark needs one adapter and no metric changes |

**GPU budget** (reference timings from the authors' files; the 3× T4 slowdown is an assumption to replace in Phase 1)

| Step | Hours |
| --- | --- |
| RAG one round, 500 | 0.7 |
| RAG two rounds, 500 | 1.3 |
| SimpleMem replay, 500 | 1.2 |
| Source augmentation, 500 | 1.2 |
| Own SimpleMem builds, 10 | 7.0 |
| Medication build, 30 patients | 3.3 |
| Medication answering, 360 questions × 3 systems | 2.7 |
| Own index, as a contingency if embeddings can't be reused | 1.7 |
| Judge and smoke runs | 3.0 |
| **Total** | **About 22** |

Kaggle's weekly quota is [commonly reported at about 30 hours](https://gpuperhour.com/blog/free-cloud-gpus-and-credits), but Kaggle doesn't publish a number. Check your account before relying on it.

| Phase | Planned GPU hours | Largest item |
| --- | --- | --- |
| 1 | About 5 | RAG runs, smoke tests, and judging |
| 2 | About 11 | Own builds |
| 3 | About 2 | Source augmentation and judging |
| 4 | About 7 | Medication build and answering |

At this pace, no calendar week should get near the quota, even if Phases 1 and 2 overlap. Check remaining quota at the start of each phase. If the measured total passes 35 hours, take the scope ladder.

## 6. Phases and gates

Effort assumes 4–6 focused hours a day. Every change must also pass the universal gate in [DONE.md](DONE.md). At the end of each phase, record the gate result, GPU hours used, and any scope change in a short note at the bottom of `REVIEW.md`.

```text
Phase 0: Audit without GPU                     days 1–3    needs: nothing
Phase 1: Own RAG on 500                        days 4–10   needs: Phase 0
Phase 2: SimpleMem fidelity and replay         days 11–14  needs: Phase 1
Phase 3: Tax causes and source augmentation    days 15–19  needs: Phase 2
   >>> Minimum shippable project ends here <<<
Phase 4: Medication-history benchmark          days 20–25  needs: Phase 3
Phase 5: Hardening and release                 days 26–30  needs: Phase 3 or 4
```

### Phase 0: Audit without GPU

Goal: prove the metric code matches the paper before spending any GPU time.

- [ ] Create the repo, `pyproject.toml`, `uv.lock`, CI, control documents, and pre-commit.
- [ ] Vendor the authors' `common/qa_prompts` and `common/judge` at `179e6a7`.
- [ ] Download the pinned dataset, then only these authors' folders: `res_simplemem_longmemeval_s_4b`, `naive_baselines/res_rag_hybrid_{1rd,2rd}_longmemeval_s_4b`.
- [ ] Inspect `memories.tar.gz` and record its format. Extract with `tarfile`'s `filter="data"`.
- [ ] Write `schema.py`, `data.py`, and `metrics.py`, with tests.
- [ ] Calculate accuracy, tax, benefit, CIs, and McNemar from the authors' verdicts.
- [ ] Run `scripts/check_repo.py` in CI.

**Gate G0:** SimpleMem 60.2%, one-round RAG 70.2%, two-round RAG 72.2%; tax 18.2 ± 0.1 and benefit 8.2 ± 0.1. Tax comes from Figure 13, whose text labels I matched to bars using Table 4's 27B value, and benefit follows from Table 1. If only those two differ, inspect before assuming a bug.

### Phase 1: Own RAG on 500

Goal: show your RAG reproduces theirs. Build it on CPU against a fake model server first, so Kaggle time isn't spent debugging logic.

- [ ] Write `chunking.py` from Appendix A.2. Compare with the authors' `buffer` chunks on all 500.
- [ ] Write the `llm.py` client and cache, with accounting per operation, and count failed calls in the manifest.
- [ ] Write `tests/fake_server.py` and run the whole pipeline on a 4-question fixture with no GPU.
- [ ] Write `kaggle_serve.sh`. Smoke-test Qwen3.5-4B, embedding, and reranking on T4 x2.
- [ ] Confirm no `<think>` text appears in answers. A [vLLM issue](https://github.com/vllm-project/vllm/issues/35574) reports `enable_thinking: false` being ignored in one image.
- [ ] Measure tokens per second and update `SLOWDOWN` in `plan_numbers.py`.
- [ ] Embed the dev 100 yourself and compare with the authors' `.npz` files. If they match, reuse theirs.
- [ ] Write `retrieval.py`, `systems/rag.py`, `answer.py`, `judge.py`, `pipeline.py`, and `cli.py`.
- [ ] Set up the judge. Judge the authors' RAG answers and compare with their verdicts.
- [ ] Run No Context, Gold Context, and RAG (one and two rounds) on dev 100, then all 500.
- [ ] Log session-level recall so retrieval bugs show up separately from answer bugs.
- [ ] Repeat 20 dev questions with the cache off, and report how often verdicts change.

**Gate G1:**

- Chunks match the authors' on at least 99% of histories.
- Judge agreement with authors' verdicts is at least 90% on their answers.
- Own one-round RAG is within ±3 points of 70.2%, and verdict agreement is at least 85%.
- No `<think>` text appears in answers.
- Unresolved failed calls and unparsable verdicts stay at or below 1%.
- At least 95% of verdicts stay the same in the repeat run.

The ±3, 85%, 1%, and 95% thresholds are my estimates. Set them in config now, not after seeing results. Two-round RAG isn't gated: the paper doesn't publish its sufficiency prompt in the appendix, so yours may differ. Use 72.2% only as a sanity check, and log the prompt in `DECISION.md`.

### Phase 2: SimpleMem fidelity and replay

Goal: show you can run SimpleMem, then replay the authors' stores through your own answer pipeline.

- [ ] Install SimpleMem at the pinned commit and connect it to your endpoint.
- [ ] Pick 10 dev histories, stratified by question type, with seed 42.
- [ ] Build 2 histories, measure the rate, and project the other 8. Apply the fallback, 5 builds, if the projection exceeds 10 hours.
- [ ] Compare store units and store tokens with the authors' for the same histories.
- [ ] Write `systems/simplemem.py`, with a live mode and a replay mode.
- [ ] In replay mode, read `retrieved_ids` with `ast.literal_eval`, load units from the stores, and render with SimpleMem's `format_contexts`.
- [ ] Replay all 500 and judge them.
- [ ] Calculate Retrieval Scope for RAG and replay, and compare with the paper's method in Appendix A.4.
- [ ] Run H1, with tax and benefit against your own RAG.

**Gate G2:**

- At least 7 of 10 own builds match the authors' verdict.
- Median store size is within ±20% of the authors'.
- Replay accuracy is within ±3 points of 60.2%, with verdict agreement of at least 85%.

Ten builds can only catch a large failure. They can't prove your builds match the authors' stores, so the report must say H1 and H2 rest on their stores.

### Phase 3: Tax causes and source augmentation

Goal: explain the tax and test the fix.

- [ ] **Faithfulness:** compare authors' gold-raw and gold-unit verdicts from `res_with_oracle_mem`, with no GPU.
- [ ] **Faithfulness, by hand:** label 100 sampled turns as no loss, partial loss, major loss, or no unit. Report it next to Table 2.
- [ ] **Accessibility:** rank gold evidence by retrieval round for raw chunks and memory units. Use the unit IDs in the oracle results; if they aren't there, use a dense-similarity rank and record the change.
- [ ] **Similarity drift:** compare query similarity with gold turns and with gold units.
- [ ] **Source augmentation:** pool authors' retrieved units with your raw candidates, jointly rerank, and fill a token budget.
- [ ] Choose the budget on dev 100 only: baseline SimpleMem median, or baseline plus RAG median. Ties go to the smaller.
- [ ] Record the candidate pools and budget rule in `DECISION.md`; the paper doesn't give these details ([D-019](DECISION.md)).
- [ ] Freeze the config, tag `prereg-h2`, then run the held-out 400 once.
- [ ] Run H2 with Holm correction. Write 5 tax and 5 benefit case studies.

**Gate G3:** identity holds exactly, H2 has a paired CI from a frozen config, and any held-out rerun is logged as a deviation. A null result passes.

### Phase 4: Medication-history benchmark

Goal: add new evidence in a domain where lost details matter. Treat it as a pilot.

- [ ] Write hypotheses H3–H5 and their tests in `extension/medmem/README.md`, then tag `prereg-med`.
- [ ] Build `generate.py` with a seed: 30 fictional patients, 12 questions each, about 25,000 tokens per history.
- [ ] Use question types: exact dose, latest change, passing allergy, stopped medicine, and time order.
- [ ] Place generated evidence among unrelated LongMemEval filler sessions.
- [ ] Store true answers and evidence turns. Assert each fact survives any LLM paraphrase, or discard that paraphrase.
- [ ] Review 20 generated histories by hand.
- [ ] Add it as a `Benchmark` adapter. Build SimpleMem, then run RAG, SimpleMem, and source augmentation.
- [ ] Calculate critical-fact survival by code, and compare exact-match scoring with the judge.
- [ ] Publish with a dataset card stating that the data is synthetic and is not clinical guidance.

**Gate G4:** every question traces to its evidence, the same seed produces identical files, CIs use patient-level resampling, and the report calls the result a pilot.

### Phase 5: Hardening and release

Goal: let a stranger verify everything without a GPU.

- [ ] Make `make reproduce` rebuild all tables and figures from published per-question results.
- [ ] Publish results with a dataset card and pinned revision.
- [ ] Add a Hugging Face Space that shows RAG and SimpleMem answers side by side from saved results.
- [ ] Write the 4–6 page report: results, gaps from the paper, likely causes, limits.
- [ ] Add `CITATION.cff`, licences, and `v1.0.0`. Credit the authors' CC-BY-4.0 results, LongMemEval, SimpleMem, and the model licences in `THIRD_PARTY.md`.
- [ ] Ask someone else to follow the README on a clean Kaggle CPU notebook.

**Gate G5:** clean-room reproduction passes without a GPU, with a tagged release and a working demo link.

## 7. Scope ladder if time or quota runs short

Cut from the top. Don't cut the fidelity gates or held-out rule.

1. Drop the Space.
2. Drop similarity drift and hand labels; keep authors' oracle results.
3. Reduce medication data to 15 patients, and report it as a smaller pilot.
4. Reduce own builds from 10 to 5.
5. Stop at Phase 3 and release. This is still a complete project.

## 8. Top risks

| Risk | Early signal | Response |
| --- | --- | --- |
| Qwen3.5 doesn't run on Kaggle's vLLM or driver | Phase 1 smoke test fails | Use the llama.cpp fallback; judge agreement must still pass G1 |
| 27B judge won't fit on T4s | Out-of-memory error | Use the D-009 fallback, and report it |
| Authors' stores can't be loaded | Phase 0 inspection fails | Open an issue with the authors. Cut H1/H2 replay to dev-set builds, and report reduced power |
| Own builds are far slower than expected | G2 projection above 10 hours | Reduce to 5 builds |
| H2 is null | Result | Report it; it is still a useful result on a smaller model |
| Authors release harness code midway | README to-do ticks | Keep your implementation; compare and log differences |
| Kaggle quota is lower than reported | Account shows under 20 hours | Spread over more weeks, or use the scope ladder |

## 9. Resume entry, after real results

> **Memory Tax Replication and Medication-History Benchmark** | Python, BM25, vLLM, Qwen, Kaggle
>
> - Replicated He et al. (2026) on 500 LongMemEval-s questions, measuring [X]% RAG vs [Y]% SimpleMem accuracy and a [Z]% memory tax.
> - Tested source augmentation on a held-out 400-question set: [W] points (95% CI [a, b], McNemar p = [p]).
> - Built a [N]-question synthetic medication-history benchmark and resumable evaluation harness, finding [one-line result].

In interviews, say what you reused: SimpleMem, the authors' stores, chunks, embeddings, and prompts. Say what you built: RAG, harness, metrics, source augmentation, analysis, and the benchmark.

## 10. Project documents

| File | Purpose |
| --- | --- |
| [CLAUDE.md](CLAUDE.md) | Agent rules, including the commenting policy |
| [DECISION.md](DECISION.md) | Every meaningful decision and its reasoning |
| [FLOW.md](FLOW.md) | How execution travels through the code |
| [BUGS.md](BUGS.md) | Start-to-finish trace of each bug and feature |
| [ARCHITECTURE.md](ARCHITECTURE.md) | High-level system map |
| [DONE.md](DONE.md) | Commands and expected outputs before work counts as done |
| [ROLLBACK.md](ROLLBACK.md) | How to undo changes |
| [PREREQUISITE.md](PREREQUISITE.md) | What to learn first |
| [REVIEW.md](REVIEW.md) | Project manager and researcher review |
| [AGENT_PROMPT.md](AGENT_PROMPT.md) | Prompt that starts a coding agent on Phase 0 |
