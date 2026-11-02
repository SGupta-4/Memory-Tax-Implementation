# Decision log

Newest entries go at the bottom. Don't edit a past decision; add a new one that supersedes it.

**Template**

```markdown
## D-NNN: Title
- **Date, phase, status:** YYYY-MM-DD, Phase N, accepted | superseded by D-NNN
- **Decision:** What was chosen.
- **Why:** The reasoning and evidence.
- **Alternatives:** What else was considered, and why not.
- **Tradeoff accepted:** What this makes worse.
- **Revisit if:** What would change this.
```

Seeded entries were decided while planning on 2026-11-02.

## D-001: Run Qwen3.5-4B and SimpleMem first

- **Decision:** Use Qwen3.5-4B as the answer model, and SimpleMem as the only memory system.
- **Why:** 4B fits one T4 at about [5 GB INT8 or 10 GB FP16](https://www.spheron.network/tools/gpu-recommender/Qwen/Qwen3.5-4B/). SimpleMem has an open repo, an MIT licence, published source-augmentation results, and a large tax on 4B.
- **Alternatives:** Qwen3.6-27B matches Table 4 but would need a separate serving setup for every answer. Running all 11 systems exceeds the quota.
- **Tradeoff accepted:** The paper's Table 4 has no 4B source-augmentation number to compare with. This makes H2 new evidence, not a direct replication.
- **Revisit if:** A stronger free GPU becomes available.

## D-002: Use the original LongMemEval, not the cleaned version

- **Decision:** Use `xiaowu0162/longmemeval`, file `longmemeval_s`, revision `2ec2a557f3`.
- **Why:** The authors' `bench_readme.md` and `download.py` both pin it. Per-question comparison needs the same data.
- **Alternatives:** `longmemeval-cleaned` removes noisy sessions and may be better benchmark data, but it breaks comparison.
- **Tradeoff accepted:** Known noisy sessions stay in the data.
- **Revisit if:** A later extension compares results across both versions.

## D-003: Evaluate on 500, with a dev and held-out split

- **Decision:** Use the authors' 100-question subset for development, and the other 400 as held-out for H2. Run H1 on all 500.
- **Why:** Authors' notes say Qwen runs used all 500, and the subset was for GPT-5.6 Luna. At n=100, power for the 10-point gap is only 0.42.
- **Alternatives:** Running only 100 is cheaper but likely to miss real effects.
- **Tradeoff accepted:** More judge calls and more care with held-out discipline.
- **Revisit if:** Answering costs far more than budgeted.

## D-004: Reuse authors' SimpleMem stores; own builds only check fidelity

- **Decision:** Replay all 500 from the authors' stores. Build only 10 yourself.
- **Why:** Sampled cost files show about 250,000 generated tokens per history. 100 builds take 27.6 hours on the authors' GPUs. The 500 stores are a 105 MB download.
- **Alternatives:** Building more histories would be purer, but needs paid compute or months of quota.
- **Tradeoff accepted:** H1 and H2 depend on stores you didn't build, so G2 must show your builds behave similarly. State this in the report.
- **Revisit if:** G2 fails, or stores can't be loaded.

## D-005: Verify, then reuse authors' chunks and embeddings

- **Decision:** Write the chunker from the paper and compare it to the authors' chunks. Embed dev 100 yourself and reuse their `.npz` files for the rest if they match.
- **Why:** This tests your code exactly and saves about 1.7 GPU hours.
- **Alternatives:** Re-embedding everything adds little evidence once dev 100 matches.
- **Tradeoff accepted:** A dependency on their published files; revisions are pinned in config.
- **Revisit if:** Cosine similarity on dev 100 is below 0.999.

## D-006: Vendor authors' prompts at a pinned commit

- **Decision:** Copy `common/qa_prompts` and `common/judge` at `179e6a7` into `vendor/memorytax/`, with its MIT licence.
- **Why:** Using identical prompts removes a source of difference and avoids transcription errors.
- **Alternatives:** Retyping from the PDF risks small errors. A git submodule adds friction on Kaggle.
- **Tradeoff accepted:** Updates need a manual copy.
- **Revisit if:** Authors fix a prompt bug.

## D-007: Write RAG and the harness yourself

- **Decision:** Implement RAG, the harness, and metrics from the paper.
- **Why:** The authors' folders for these are placeholders at `179e6a7`, and this is the project's core portfolio evidence.
- **Alternatives:** Waiting for their release adds risk and removes your contribution.
- **Tradeoff accepted:** More work and a risk of differences, which G1 catches.
- **Revisit if:** They publish a harness: compare it, and log what differs.

## D-008: Use vLLM first, llama.cpp as fallback

- **Decision:** Serve with vLLM. Fall back to llama.cpp if the Phase 1 smoke test fails.
- **Why:** The authors' clients expect OpenAI-compatible endpoints, and vLLM batches parallel requests well.
- **Alternatives:** Hugging Face `transformers` would be simpler, but slower across parallel requests.
- **Tradeoff accepted:** Quantized fallback answers can differ from FP16 answers, so G1 must pass again.
- **Revisit if:** The smoke test fails.

## D-009: Judge with Qwen3.6-27B if possible, otherwise calibrated 4B

- **Decision:** Try a 4-bit [Qwen3.6-27B GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) across both T4s in a separate session. If it doesn't fit, use Qwen3.5-4B as judge.
- **Why:** The authors' judge was 27B. Judge calls are short, so low speed is acceptable.
- **Alternatives:** A paid API would break the free constraint. The authors' verdicts can't judge your new answers.
- **Tradeoff accepted:** Quantized or smaller judges may differ. Accept one only if it agrees at least 90% with the authors' verdicts on their answers. Hand-check 50 disagreements, and report both figures.
- **Revisit if:** Neither judge reaches 90%. Stop and ask.

## D-010: Use SQLite for the call cache

- **Decision:** Cache model calls in one SQLite file, keyed by a hash of model, revision, prompt version, messages, and settings.
- **Why:** SQLite is in the standard library, writes safely, handles concurrent async writes in WAL mode, and uploads as one file.
- **Alternatives:** JSONL shards need custom locking and lookups. Redis is another service to run on Kaggle.
- **Tradeoff accepted:** Binary file; a small export script makes it inspectable.
- **Revisit if:** Runs span several machines at once.

## D-011: Use brute-force NumPy search; no vector database

- **Decision:** Keep embeddings per history in `.npz`, and search with NumPy dot products.
- **Why:** The sampled history has 454 chunks, so exact search is simple and fast. Exact search also removes an approximation from the comparison.
- **Alternatives:** FAISS or Qdrant solve a scale problem this project doesn't have.
- **Tradeoff accepted:** Not suitable for millions of chunks.
- **Revisit if:** One index goes above about 100,000 vectors.

## D-012: No tracking server, orchestrator, or Docker

- **Decision:** Use a manifest, JSONL, and `make`. Skip MLflow, Airflow, and Docker.
- **Why:** Kaggle already provides the runtime. Immutable runs with manifests answer what ran and with which inputs.
- **Alternatives:** MLflow helps teams with many live runs. This project has one developer and a few dozen runs.
- **Tradeoff accepted:** No hosted dashboard.
- **Revisit if:** A team joins, or runs move off Kaggle.

## D-013: Keep exactly two extension interfaces

- **Decision:** Define `MemorySystem` and `Benchmark` protocols.
- **Why:** There are already 3 systems (RAG, SimpleMem, source augmentation) and 2 benchmarks. This makes adding another a one-file change.
- **Alternatives:** More plugin layers solve hypothetical problems.
- **Tradeoff accepted:** A very different memory form may need a small interface change.
- **Revisit if:** A new system needs state that doesn't fit `build` and `retrieve`.

## D-014: Use paired tests and bootstrap CIs

- **Decision:** Use exact McNemar, 10,000-sample bootstrap CIs, and Holm across H1 and H2.
- **Why:** Systems answer the same questions, so results are paired binary outcomes.
- **Alternatives:** An unpaired test wastes the pairing. Reporting only point estimates hides uncertainty.
- **Tradeoff accepted:** Slightly more analysis code.
- **Revisit if:** A hypothesis needs clustered data, as in D-015.

## D-015: 30 patients with 12 questions each; resample by patient

- **Decision:** Generate 360 questions from 30 histories, and resample by patient.
- **Why:** Build cost depends on histories, not questions, so more questions per history is cheap. Questions in one history are correlated, so resampling must respect that.
- **Alternatives:** One question per history gives independence but costs far more GPU time.
- **Tradeoff accepted:** Lower effective sample size. Report it as a pilot.
- **Revisit if:** Results are promising enough to justify a larger set.

## D-016: Stay free

- **Decision:** Use only free compute and free hosting.
- **Why:** Project constraint. The budget of about 24 GPU hours appears to fit.
- **Alternatives:** A rented GPU would speed up builds, but isn't required.
- **Tradeoff accepted:** Slower runs and session limits.
- **Revisit if:** You choose to pay to finish faster. This needs your approval.

## D-017: Pin SimpleMem to the paper's commit

- **Decision:** Use `db80b6a7c5`, with no local changes if possible.
- **Why:** The authors' `setup.sh` pins this commit. The repo's `main` has since added multimodal and self-evolving code.
- **Alternatives:** `main` may be better, but would measure a different system.
- **Tradeoff accepted:** Possible old-dependency conflicts; isolate its environment.
- **Revisit if:** It can't run on Kaggle. Log any patch here.

## D-018: Separate answering and judging stages

- **Decision:** Answer in one session and judge in another.
- **Why:** The 27B judge needs both GPUs, and answering uses them too. Verdicts can also be redone without new answers.
- **Alternatives:** Judging inline is simpler, but would force a smaller judge or constant model swapping.
- **Tradeoff accepted:** Results appear only after a second session.
- **Revisit if:** A 4B judge is chosen; it could then run inline.

## D-019: Define source augmentation where the paper doesn't

- **Decision:** Pool the authors' retrieved units for each question with the BM25 and dense raw candidates, jointly rerank with Qwen3-Reranker-0.6B, and take results in rank order until a token budget is full. Choose between two budgets on dev 100 only.
- **Why:** The paper says only to retrieve from source history alongside memory and jointly rerank before QA. Its pool sizes and context limit aren't stated, and the harness isn't released.
- **Alternatives:** Appending raw chunks without reranking is simpler, but isn't what the paper describes.
- **Tradeoff accepted:** H2 tests this version of the idea, not necessarily the authors' exact code.
- **Revisit if:** The authors release their harness: run theirs on dev 100 as a sensitivity check, without changing H2's frozen config.

## D-020: Refuse comparisons across different judges

- **Decision:** Each verdict stores `judge_id`, and `metrics.py` refuses to pair verdicts with different IDs.
- **Why:** If your judge differs from the authors', a mixed comparison could produce a fake tax or benefit.
- **Alternatives:** Correcting for judge differences would add assumptions and make results harder to defend.
- **Tradeoff accepted:** Comparing with the authors' verdicts needs a separate audit command and is reported separately.
- **Revisit if:** Never for confirmatory results.

## D-021: Python toolchain and build backend

- **Date, phase, status:** 2026-11-02, Phase 0, accepted
- **Decision:** Python 3.11 managed by `uv`, built with `uv_build`. Runtime dependencies: `numpy`, `scipy` (exact binomial test), and `pyyaml` (configs, read with `safe_load` only). Dev group: `pytest`, `ruff`, `mypy` in strict mode, and `types-pyyaml` for mypy.
- **Why:** This is the stack `AGENT_PROMPT.md` allows. `uv_build` is uv's own backend, so there's no second build tool to learn, and it supports the `src/` layout with no configuration.
- **Alternatives:** `hatchling` or `setuptools` work equally well but add a separate tool.
- **Tradeoff accepted:** `uv_build` is younger than `hatchling`; the version is capped in `pyproject.toml`.
- **Revisit if:** A build needs features `uv_build` lacks, such as compiled extensions.

## D-022: Pre-commit with local hooks; no scipy stubs

- **Date, phase, status:** 2026-11-02, Phase 0, accepted
- **Decision:** Add `pre-commit` to the dev group, with only `repo: local` hooks that call the universal gate's own commands through `uv run`. For mypy, mark `scipy.*` as untyped with an override instead of adding `scipy-stubs`.
- **Why:** Phase 0 asks for pre-commit. Local hooks fetch nothing from third-party hook repos and can't drift from CI. Only one scipy function is used, and its result is wrapped in `float()`, so stubs would add a dependency for one call.
- **Alternatives:** Upstream hook repos (`ruff-pre-commit`, `mirrors-mypy`) pin their own tool versions, which can disagree with `uv.lock`.
- **Tradeoff accepted:** scipy calls aren't type-checked.
- **Revisit if:** `metrics.py` uses more of scipy.

## D-023: Verdict loader accepts fixed field names and rejects unparsable verdicts

- **Date, phase, status:** 2026-11-02, Phase 0, accepted; recheck when the authors' files are inspected (B-004)
- **Decision:** `data.load_verdicts` reads every `.json` and `.jsonl` under a folder and takes any object with a `question_id` and one of `verdict`, `judge_verdict`, `label`, `judge_label`, `correct`, `is_correct`, first match winning. Values must be 0/1, a boolean, or `CORRECT`/`WRONG`. A null or other value, a second verdict for the same question, or no verdicts at all raises an error naming the file, question, and keys seen.
- **Why:** huggingface.co was blocked in the session that wrote the loader, so the authors' layout couldn't be inspected. Failing loudly on the first real run is safer than guessing silently. The vendored `parse_verdict` returns `None` for unparsable replies and says they're "counted separately", so the policy for them is a research decision, not a loader default.
- **Alternatives:** Treating null as wrong would make G0 pass or fail on an unrecorded assumption.
- **Tradeoff accepted:** The first real run may need a one-line loader change, recorded here.
- **Revisit if:** The authors' files use another field, or contain null verdicts. Then add a decision on how the paper counted them.

## D-024: Phase 0 builds only what the audit uses

- **Date, phase, status:** 2026-11-02, Phase 0, accepted
- **Decision:** `schema.py` defines only `Question`, `Turn`, and `Verdict`. `Retrieved`, `Answer`, `Manifest`, and the `Benchmark` protocol arrive in Phase 1, with the stages that use them. Holm correction and patient-level bootstrap arrive in Phases 3 and 4.
- **Why:** Phase 0 is an audit of verdicts. Unused types would be designed before their first caller exists.
- **Alternatives:** Writing the full schema now matches the module table, but guesses fields Phase 1 will decide.
- **Tradeoff accepted:** `ARCHITECTURE.md` lists modules that are partly built; it now says so.
- **Revisit if:** Never; Phase 1 adds them.

## D-025: How the audit applies G0

- **Date, phase, status:** 2026-11-02, Phase 0, accepted
- **Decision:** Accuracies must equal the published one-decimal values (60.2, 70.2, 72.2) after rounding. Tax and benefit may differ by up to 0.1 points. n must be 500. Question ids in every run must equal the dataset's 500 ids. The audit prints `G0 PASS` or `G0 FAIL: <reasons>` and exits 1 on failure.
- **Why:** The plan gives a tolerance only for tax and benefit. With n=500, a difference of one question moves accuracy by 0.2 points, so exact one-decimal agreement means the same correct counts. Checking ids is how B-001 is verified.
- **Alternatives:** A shared ±0.1 on accuracies would allow the same thing, as 0.1 is below one question; the exact form is clearer.
- **Tradeoff accepted:** None; this restates the preregistered gate in code.
- **Revisit if:** Never without a human; gate thresholds are preregistered.

## D-026: Authors' verdict layout confirmed; audit reads judge.jsonl only

- **Date, phase, status:** 2026-11-02, Phase 0, accepted; narrows D-023
- **Decision:** `memtax audit` reads `<run>/judge_res/judge.jsonl` instead of walking each run folder. Authors' results are pinned to `MemoryTax-Res-Qwen3-5-4B` revision `987bfe5fb8317900abbad3a3260672299db1229c`.
- **Why:** The repo README and the files show one JSON line per question: `question_id`, `category`, `verdict` (0/1), `raw`, `gold_used`, `prompt`, and `judge_prompt_version` (`judge/unified-v1` everywhere). `verdict` is D-023's first key, and none of the 1,500 verdicts is null, so no policy for unparsable verdicts is needed. `qa_res/` also carries `question_id`, so walking it adds risk and nothing else.
- **Alternatives:** Keep walking whole folders. That works today, but would silently pick up any future file with a matching key.
- **Tradeoff accepted:** A layout change in the authors' repo needs a path change here; the revision pin prevents surprise changes.
- **Revisit if:** The pinned revision changes.

## D-027: Format of the authors' SimpleMem stores (`memories.tar.gz`)

- **Date, phase, status:** 2026-11-02, Phase 0, accepted
- **Decision:** Extract `res_simplemem_longmemeval_s_4b/memories.tar.gz` (revision `987bfe5f`, sha256 `8367f3c2…b643`) with `tarfile` and `filter="data"` into `data/stores/`. Phase 2's replay mode reads the format below.
- **Format, as inspected:** 1,500 regular files (485 MB), with no links and no absolute or `..` paths, under `memories/<question_id>/`. Each of the 500 questions, including the `_abs` ones, has three files:
  - `memory_entries.json`: `{question_id, n_entries, entries}`. Each entry has `entry_id` (UUID), `lossless_restatement`, `keywords`, `timestamp`, `location`, `persons`, `entities`, and `topic`. Stores hold 1,057 to 1,917 entries; the median is 1,448.
  - `provenance.jsonl`: one `{entry_id, src_dids}` per entry, linking a unit to the turns it came from.
  - `did_map.json`: turn id -> `[n, session_index, turn_index]` into `haystack_sessions`. All 246,918 indices are valid. `n` equals `len(content) // 3` in 246,886 cases, and the 32 exceptions aren't explained yet. 493 of 500 maps cover every turn.
- **Why it matters:** All 27,987 ids in the SimpleMem `qa_res` `retrieved_ids` (parsed with `ast.literal_eval`) resolve to `entry_id`s in these stores, so replay can rebuild every memory block. `provenance` plus `did_map` link units back to gold turns, which Phase 3 accessibility needs.
- **Alternatives:** None; this records what the files are.
- **Tradeoff accepted:** The meaning of `n` is inferred, not documented; Phase 3 must confirm it before relying on it.
- **Revisit if:** The 32 `n` mismatches or the 7 partial maps matter for Phase 3 accessibility.

## D-028: Chunking rules as matched against the authors' buffer chunks

- **Date, phase, status:** 2026-11-02, Phase 1, accepted
- **Decision:** `chunking.py` implements Appendix A.2 with the details the paper leaves out, each found by matching the authors' `buffer` chunks (`MemoryTax-Res-Other` at `00968a85`):
  - The header is `(session date-time: <date>)\n`. Turns are rendered as `[role: <role>]: <text>` and joined by `\n`.
  - Each turn's text is stripped, and empty turns are skipped.
  - Sizes count the joined role lines, not the header.
  - A line of 1,024 characters or more stands alone. Short lines merge while the joined length stays at or under 1,024.
  - On overflow, the last turn carries over only when the flushed chunk merged more than one turn. If the carried turn still can't fit with the next one, it is emitted alone.
  - Lines over 8,192 characters split into slices `[s, s+8192)` every 7,168 characters, each stripped, with only the first keeping the role prefix.
  - Doc ids are `<question_id>|d<n>`, counted in history order.
- **Why:** `memtax check-chunks --against authors` gives 500/500 identical histories (text, turn ids, and doc ids), so the authors' embeddings line up row for row (D-005).
- **Alternatives:** A literal reading of the paper (sizes counting text only, overlap after every flush, `> 1024` for long turns) matched 0 of 60 histories.
- **Tradeoff accepted:** Some rules, like the overlap quirk, reproduce their implementation rather than an obviously better design.
- **Revisit if:** Never for the replication; the medication benchmark reuses the same rules.

## D-029: Retrieval details the paper leaves open

- **Date, phase, status:** 2026-11-02, Phase 1, accepted; BM25 differs from the authors' (B-006)
- **Decision:**
  - **BM25:** Okapi BM25 with Lucene's idf `log(1 + (N - df + 0.5) / (df + 0.5))`, k1 1.5, b 0.75, and lowercase `\w+` tokens, over each history's chunks.
  - **Dense:** cosine similarity of the query vector with every chunk vector.
  - **Ties:** broken by chunk order.
  - **Queries:** embedded with Qwen3-Embedding's standard instruction, `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`.
  - **Reranking:** uses the Qwen3-Reranker model card template and its default instruction, served by vLLM's `/rerank`.
- **Why:** The paper names the models but not these settings. Dense matches the authors' to about 0.001 cosine on their own vectors. Their query file records `query_prompt: true`, which fits the standard instruction; `memtax check-embeddings` confirms it on Kaggle.
- **Alternatives:** A library such as `rank_bm25` or `bm25s`. A search over tokenizers, idf formulas, k1, and stopwords found no variant that reproduces the authors' BM25 scores (B-006), so a dependency buys nothing.
- **Tradeoff accepted:** Our candidate pools differ slightly from the authors', so G1's ±3-point and 85% agreement checks carry that difference.
- **Revisit if:** G1's RAG accuracy or agreement fails. Then try stemming next (B-006).

## D-030: Pieces we define because the paper doesn't publish them

- **Date, phase, status:** 2026-11-02, Phase 1, accepted
- **Decision:**
  - **Sufficiency prompt:** our own, `memtax/sufficiency-v1` in `systems/rag.py`. It asks for `{"sufficient", "why", "next_query"}`, the same fields the authors' two-round records log. An unparsable reply stops after round one, and an empty `next_query` falls back to the question. Round two searches only the chunks round one didn't pick.
  - **RAG memory block:** chunk texts joined by blank lines.
  - **Gold Context:** renders `has_answer` turns as `(<date>) [role: r]: text`, the authors' evidence format.
- **Why:** None of these is in the paper or the released code. Two-round RAG isn't gated for this reason (plan section 6).
- **Alternatives:** Copying the sufficiency prompt from another system would still be a guess.
- **Tradeoff accepted:** Two-round results and some RAG-versus-authors differences may come from these choices.
- **Revisit if:** The authors release their harness.

## D-031: How G1 is computed, set before any run

- **Date, phase, status:** 2026-11-02, Phase 1, accepted (preregistered with `configs/gates.yaml`)
- **Decision:**
  - **RAG accuracy:** a missing or unparsable verdict counts as wrong.
  - **Agreements:** computed over parsable verdicts, while the failure-rate check caps how many are missing or unparsable.
  - **Judge agreement:** judged on the authors' dev-100 one-round RAG answers (`pred`), with no question left unjudged.
  - **Repeat check:** `configs/kaggle/rag1_repeat20.yaml`, the first 20 dev questions by id with the cache off, compared with `rag1_all` verdicts.
  - **Embedding reuse (D-005):** the authors' vectors are reused only if mean row cosine is at least 0.999 for both dev-100 corpus vectors and query vectors.
- **Why:** The plan lists the thresholds but not these accounting rules, and they must be fixed before results exist.
- **Alternatives:** Excluding unparsable verdicts from accuracy would hide judge failures inside a smaller denominator.
- **Tradeoff accepted:** A run with failures scores slightly lower than its parsable verdicts suggest.
- **Revisit if:** Never after results, without a human (CLAUDE.md).

## D-032: Harness engineering choices

- **Date, phase, status:** 2026-11-02, Phase 1, accepted
- **Decision:**
  - **HTTP:** standard-library `urllib`, run in worker threads behind an `asyncio` semaphore.
  - **Corpus embeddings:** stored as `.npz` per history under `data/index/<model>@<revision>/`, not in the SQLite cache. Query, chat, and rerank calls are cached.
  - **Run ids:** `run_id` hashes only settings that change results (dataset hash, subset, system, model and revision per role, `limit`, `use_cache`), not paths, base URLs, or concurrency.
  - **Retries:** a failed question is not written, so rerunning the same command retries it.
  - **Crash recovery:** a crash-truncated final JSONL line is cut off on the next read.
  - **Judge retry:** a retry of an unparsable verdict carries `attempt` in its cache key.
- **Why:** No new dependency is needed for four endpoints. Cached vectors would bloat the cache to gigabytes. Resuming from another Kaggle session must reach the same run.
- **Alternatives:** `httpx` or `aiohttp` (a new dependency for little gain); hashing the whole config (a new run on every path change).
- **Tradeoff accepted:** Threads, not true async I/O; fine at concurrency 8.
- **Revisit if:** Concurrency needs to go far above about 32.
