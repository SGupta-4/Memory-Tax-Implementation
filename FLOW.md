# Execution flow

This file shows how execution actually travels between files and functions. It describes the planned design until Phase 1 builds it. Update it in the same commit as any call-path change.

## Currently modifying

Keep this current. Clear it when the change lands.

| Field | Value |
| --- | --- |
| Task | Phase 0: G0 audit waiting on the dataset file; large-file hosts blocked (B-004) |
| Bug or feature | F-001 in `BUGS.md` |
| Files on the path | `src/memtax/data.py`, `src/memtax/cli.py` |
| Entry point affected | `memtax audit --bench longmemeval_s` |
| Downstream that could break | `tests/test_metrics.py` real-data test; G0 output in `BUGS.md` F-001 |

## Entry points

Built so far: `memtax audit`, `memtax smoke`, and `python scripts/check_repo.py`. The rest are planned.

| Command | Purpose | GPU? |
| --- | --- | --- |
| `memtax run --config C --stage S [--shard i/n]` | Run one stage | Depends on stage |
| `memtax metrics --system A --reference B` | Paired statistics | No |
| `memtax audit --bench longmemeval_s` | Statistics on authors' verdicts | No |
| `memtax smoke --config configs/fixture.yaml` | Whole pipeline on a fixture and fake server | No |
| `memtax check-chunks --against authors` | Compare chunks with authors' | No |
| `memtax judge-agreement --dev100` | Compare your judge with authors' verdicts | Yes |
| `memtax gate N` | Check a phase gate and print results | No |
| `make reproduce` | Rebuild tables from published results | No |
| `python scripts/check_repo.py` | Check control documents and docstrings | No |
| `python -m extension.medmem.generate --seed 42` | Generate medication data | Only for paraphrase |

## Phase 0 audit path

```text
cli.main(["audit", "--bench", "longmemeval_s"])
└─ cli.audit(data_dir)
   ├─ data.load_questions(longmemeval/longmemeval_s.json)    validates fields; ids unique
   ├─ data.load_verdicts(folder, AUTHORS_JUDGE_ID)          for simplemem, rag1, rag2 (D-023)
   ├─ cli._check_ids(...)                                   each run's ids == dataset ids (B-001)
   ├─ metrics.pair(system, rag1)                            same judge_id, same question set
   ├─ metrics.paired_summary(...)                           counts; raises if the identity fails
   ├─ metrics.bootstrap_ci(...), metrics.mcnemar_exact(...)
   └─ print, then "G0 PASS" or "G0 FAIL: ..." (D-025)
```

`memtax smoke --config configs/fixture.yaml` runs the same load, pair, and summary steps on the 4-question fixture in `tests/fixtures/`.

## Phase 0 data layout

`memtax audit` reads these paths under `--data-dir` (default `data/`, not committed). The authors' paths were confirmed on the Hub at revision `987bfe5f` ([D-026](DECISION.md)).

```text
data/longmemeval/longmemeval_s.json
data/authors/res_simplemem_longmemeval_s_4b/judge_res/judge.jsonl                     verdicts read by audit
data/authors/res_simplemem_longmemeval_s_4b/{qa_res,cost}/, memories.tar.gz            answers, costs, stores
data/authors/naive_baselines/res_rag_hybrid_{1rd,2rd}_longmemeval_s_4b/judge_res/judge.jsonl
```

Download with the `hf` CLI. The dataset file is named `longmemeval_s` on the Hub, with no extension:

```bash
hf download xiaowu0162/longmemeval longmemeval_s --repo-type dataset --revision 2ec2a557f339b6c0369619b1ed5793734cc87533 --local-dir data/longmemeval && mv data/longmemeval/longmemeval_s data/longmemeval/longmemeval_s.json
hf download hexuan21/MemoryTax-Res-Qwen3-5-4B --repo-type dataset --revision 987bfe5fb8317900abbad3a3260672299db1229c --include "res_simplemem_longmemeval_s_4b/*" "naive_baselines/res_rag_hybrid_1rd_longmemeval_s_4b/*" "naive_baselines/res_rag_hybrid_2rd_longmemeval_s_4b/*" --local-dir data/authors
```

## Main path: one stage

```text
cli.main()
└─ pipeline.run_stage(config, stage, shard)
   ├─ pipeline.resolve_run(config)          config hash -> run_id; writes manifest once
   ├─ data.load_benchmark(config)           returns Benchmark adapter
   ├─ pipeline.pending_questions(...)       all questions in this shard, minus finished ones
   └─ for each pending question, with bounded concurrency:
      ├─ stage == "retrieve" -> system.retrieve(question)      see below
      ├─ stage == "answer"   -> answer.answer(question, retrieved)
      ├─ stage == "judge"    -> judge.judge(question, answer)
      └─ pipeline.append_record(...)        flush one JSONL line, so a crash loses at most one
```

Each answer and judge call goes through the same route:

```text
llm.chat(messages, settings)
├─ key = sha256(model, revision, prompt_version, messages, settings)
├─ cache hit?  -> return stored reply
└─ cache miss  -> POST to server with retries -> store -> count tokens -> return reply
```

## Retrieval paths

**One-round RAG**

```text
systems/rag.RagSystem.retrieve(q)
├─ retrieval.bm25_top_k(q, k=10)
├─ retrieval.dense_top_k(q, k=10)         embeddings from index stage
├─ retrieval.union_dedup(...)
└─ llm.rerank(...) -> top 10
```

**Two-round RAG**

```text
RagSystem.retrieve(q) as above
└─ llm.chat(sufficiency prompt)
   ├─ enough -> stop
   └─ follow-up query -> retrieval over the remaining pool -> merge both rounds
```

**SimpleMem replay**

```text
systems/simplemem.SimpleMemSystem.retrieve(q)        mode = "replay"
├─ read authors' qa_results_<id>.json
├─ ast.literal_eval(retrieved_ids)
├─ load units from data/stores/<id>/
└─ upstream format_contexts(units) -> memory_block
```

**SimpleMem live**

```text
SimpleMemSystem.build(history)                       mode = "live"
└─ upstream memory builder -> llm endpoint -> data/own_stores/<id>/
SimpleMemSystem.retrieve(q)
└─ upstream retriever -> memory_block
```

**Source augmentation**

```text
systems/src_aug.SrcAugSystem.retrieve(q)
├─ SimpleMemSystem.retrieve(q)            memory candidates
├─ retrieval BM25 + dense union           raw candidates
├─ llm.rerank(q, memory + raw)            one joint ranking
└─ take in rank order until token budget -> memory_block
```

## Answer and judge

```text
answer.answer(q, retrieved)
└─ vendor qa_longmemeval.build_messages(memory_block, question_date, question, question_type)
   └─ llm.chat(temperature=0, thinking off) -> Answer

judge.judge(q, answer)
├─ vendor pick_prompt(bench, category)    GENERAL, or PREF for preference questions
├─ vendor build_messages(...)
├─ llm.chat(judge endpoint, temperature=0)
└─ vendor parse_verdict(raw) -> 0 or 1; None is logged, retried once, then flagged
```

## Metrics

```text
cli metrics
└─ metrics.load_verdicts(run_a), load_verdicts(run_b)
   ├─ refuse runs marked invalid in runs/index.csv
   ├─ assert question sets are identical and judge_id matches
   ├─ metrics.paired_summary(...)         accuracy, benefit, tax, net; asserts the identity
   ├─ metrics.mcnemar_exact(...)
   ├─ metrics.bootstrap_ci(..., seed, cluster=None | patient_id)
   └─ write metrics.json and tables
```

## Order of a full Phase 1 session on Kaggle

```text
1. scripts/kaggle_serve.sh answer     start 4B on GPU 0; embedding and reranker on GPU 1
2. Health check                       one call per endpoint; no <think> text allowed
3. memtax run --stage index
4. memtax run --stage retrieve
5. memtax run --stage answer
6. Upload run and cache to private HF dataset, at least hourly and before session end
7. New session: kaggle_serve.sh judge
8. memtax run --stage judge
9. Upload, then memtax metrics
```
