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
