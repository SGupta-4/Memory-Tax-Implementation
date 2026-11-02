# Prerequisites

Learn these before you implement the paper, so you can understand and defend the code. Skip anything you can already explain without notes. I haven't re-opened the general learning links below today, so use a web search if one has moved.

**Estimated time:** about 5–7 focused days if most of this is new, less if you already know RAG and evaluation. You don't need to finish everything before Phase 0; the "needed by" column shows when each topic matters.

## Priority 1: the core ideas

| Topic | Learn | Needed by | You're ready when you can... |
| --- | --- | --- | --- |
| The paper | Sections 1–4 and 6, Appendices A–C, Tables 1, 4, 19 | Phase 0 | Define benefit, tax, faithfulness, and accessibility, and say why RAG is the reference |
| Agent memory | Memory `I(H)`, retrieval `R(q, M)`, generation `G(q, C)`; flat, hierarchical, graph, and learned types | Phase 0 | Explain how a memory unit differs from a raw chunk |
| SimpleMem | Compression, synthesis, and intent-aware retrieval | Phase 2 | Trace how one user turn becomes a stored fact |
| LongMemEval | Question types, `has_answer`, `answer_session_ids`, abstention | Phase 0 | Explain why abstention questions have no gold turn |
| BM25 | Term frequency, inverse document frequency, length normalization | Phase 1 | Compute a BM25 score by hand for a tiny example |
| Dense retrieval | Embeddings, cosine similarity, query and document prompts | Phase 1 | Explain why dense search may miss an exact dose |
| Reranking | Cross-encoders, candidate pools, why rerank after retrieval | Phase 1 | Explain why reranking every chunk is too slow |
| Chunking | Turn boundaries, overlap, metadata headers | Phase 1 | Explain why chunks should not cross sessions |

## Priority 2: evaluation and statistics

This is where most replication mistakes happen.

| Topic | Learn | Needed by | You're ready when you can... |
| --- | --- | --- | --- |
| Paired comparison | Why same-question results aren't independent | Phase 0 | Explain why two accuracy CIs overlapping doesn't settle the comparison |
| McNemar test | Discordant pairs, exact binomial form | Phase 0 | Calculate it by hand from 10 wins and 3 losses |
| Bootstrap CIs | Resampling and percentile intervals | Phase 0 | Explain why medication data resamples patients, not questions |
| Statistical power | Effect size, sample size, what a null means | Phase 0 | Explain why a null at n=100 is weak evidence |
| Multiple comparisons | Holm correction, exploratory vs confirmatory | Phase 3 | Explain why per-type results are exploratory |
| Preregistration | Dev and held-out sets, deviations | Phase 3 | Explain why looking at held-out results before tuning invalidates the test |
| LLM judges | Calibration, agreement, judge bias | Phase 1 | Explain what 90% judge agreement does and doesn't prove |

## Priority 3: serving and systems

| Topic | Learn | Needed by | You're ready when you can... |
| --- | --- | --- | --- |
| vLLM | OpenAI-compatible server, batching, `--max-model-len`, `chat_template_kwargs` | Phase 1 | Start a model and call it with `curl` |
| GPU memory | FP16, INT8, 4-bit, KV cache, `--gpu-memory-utilization` | Phase 1 | Estimate whether a model fits on a 16 GB T4 |
| Kaggle | Sessions, secrets, saved versions, storage, GPU quota | Phase 1 | Resume a run in a new session from an HF checkpoint |
| Async Python | `asyncio`, semaphores, retries with backoff | Phase 1 | Send 100 requests with at most 8 running at once |
| SQLite | WAL mode, transactions, primary keys | Phase 1 | Explain why cache writes survive a crash |
| Hashing for caches | Stable JSON serialization and SHA-256 | Phase 1 | Explain what must be in the cache key |
| Tokenizers | Tokens vs characters; counting by model tokenizer | Phase 2 | Calculate Retrieval Scope for one question |

## Priority 4: engineering practice

| Topic | Learn | Needed by | You're ready when you can... |
| --- | --- | --- | --- |
| Git | Branches, tags, `revert`, not losing work | Phase 0 | Undo a pushed commit safely |
| Python packaging | `pyproject.toml`, `uv`, lockfiles | Phase 0 | Recreate the environment from a clean machine |
| Testing | `pytest`, fixtures, markers, tests that fail first | Phase 0 | Write a failing test for a chunking bug |
| Typing and linting | Type hints, `mypy`, `ruff` | Phase 0 | Fix a `mypy` error without using `Any` |
| CI | GitHub Actions on CPU-only tests | Phase 0 | Make CI fail on a broken test |
| Immutability and provenance | Config hashes, manifests, pinned revisions | Phase 1 | Say exactly which commit and model produced a table |
| Hugging Face Hub | `hf download --include`, dataset cards, revisions | Phase 0 | Download one folder at a pinned revision |

## Priority 5: the extension

| Topic | Learn | Needed by | You're ready when you can... |
| --- | --- | --- | --- |
| Synthetic data | Seeds, templates, leakage, truth tracking | Phase 4 | Prove a generated question's answer is in its history |
| Clustered data | Why questions per patient aren't independent | Phase 4 | Explain effective sample size |
| Dataset documentation | Dataset cards, intended use, limitations | Phase 4 | Write why the data isn't clinical guidance |
| Medication basics | Dose, frequency, stopped and changed medicines, allergy records | Phase 4 | Design a realistic history with no patient data |

## Reading order

| Day | Read or do |
| --- | --- |
| 1 | Memory Tax, sections 1–4; recreate one benefit and one tax example by hand |
| 2 | LongMemEval paper; open 3 examples and find their evidence turns |
| 3 | IR basics: BM25, embeddings, reranking; build a tiny RAG on 20 turns |
| 4 | Statistics: McNemar, bootstrap, power; run `scripts/plan_numbers.py` and change its inputs |
| 5 | SimpleMem paper; read its builder and retriever at the pinned commit |
| 6 | vLLM on Kaggle: serve a small model, call it, and stop it |
| 7 | Git, `pytest`, `uv`, CI: set up a small repo that fails CI on a broken test |

## Resources

**Project sources**

- [Memory Tax paper](https://arxiv.org/abs/2610.34073) and [project page](https://hexuan21.github.io/MemoryTax/)
- [Authors' repo](https://github.com/hexuan21/MemoryTax)
- [LongMemEval paper](https://arxiv.org/abs/2410.10813) and [repo](https://github.com/xiaowu0162/LongMemEval)
- [SimpleMem paper](https://arxiv.org/abs/2601.02553) and [repo](https://github.com/aiming-lab/SimpleMem)

**General learning**

- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/): chapters 1, 6, 8
- [Sentence Transformers documentation](https://www.sbert.net/): retrieval and reranking
- [vLLM documentation](https://docs.vllm.ai/): OpenAI-compatible server
- [Python `asyncio` documentation](https://docs.python.org/3/library/asyncio.html)
- [Pro Git](https://git-scm.com/book/en/v2): chapters 2, 3, 7
- [pytest documentation](https://docs.pytest.org/)

## Final check before Phase 0

Answer these without notes:

1. Why can a memory system have a high tax and still have positive net value?
2. If accuracy is 60.2% for SimpleMem, 70.2% for RAG, and tax is 18.2%, what's the benefit?
3. Why does the plan reuse the authors' stores, and what does that make you responsible for proving?
4. Why is the 400-question set only run once?
5. What is in the cache key, and what breaks if the model revision isn't?
6. If H2 is null at n=400, what can and can't you conclude?

**Answer to 2:** 8.2%, because net value is -10.0 and benefit equals net value plus tax.
