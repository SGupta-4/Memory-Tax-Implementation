# Execution flow

This file shows how execution actually travels between files and functions. It describes the planned design until Phase 1 builds it. Update it in the same commit as any call-path change.

## Currently modifying

Keep this current. Clear it when the change lands.

| Field | Value |
| --- | --- |
| Task | Phase 1: CPU harness built; Kaggle steps next (F-002 follow-up) |
| Bug or feature | F-002, B-005, B-006 in `BUGS.md` |
| Files on the path | `src/memtax/pipeline.py`, `src/memtax/systems/rag.py`, `src/memtax/gates.py`, `scripts/kaggle_serve.sh` |
| Entry point affected | `memtax run`, `memtax gate 1`, `scripts/kaggle_serve.sh` |
| Downstream that could break | Run ids (the config identity hash) and the cache key: changing either orphans existing runs or replies |

## Entry points

Built so far: `memtax audit`, `smoke`, `check-chunks`, `run`, `check-embeddings`, `judge-agreement`, `gate 1`, `scripts/kaggle_serve.sh` (untested on GPU), and `python scripts/check_repo.py`. The rest are planned.

| Command | Purpose | GPU? |
| --- | --- | --- |
| `memtax run --config C --stage S [--shard i/n]` | Run one stage | Depends on stage |
| `memtax metrics --system A --reference B` | Paired statistics | No |
| `memtax audit --bench longmemeval_s` | Statistics on authors' verdicts | No |
| `memtax smoke --config configs/fixture.yaml` | Whole pipeline on a fixture and fake server | No |
| `memtax check-chunks --against authors` | Compare chunks with authors' | No |
| `memtax judge-agreement --dev100` | Compare your judge with authors' verdicts | Yes |
| `memtax check-embeddings` | Own dev-100 corpus and query vectors vs the authors' (D-005) | Embedding server |
| `memtax gate N` | Check a phase gate and print results | No |
| `make reproduce` | Rebuild tables from published results | No |
| `python scripts/check_repo.py` | Check control documents and docstrings | No |
| `python -m extension.medmem.generate --seed 42` | Generate medication data | Only for paraphrase |

## Phase 0 audit path

```text
cli.main(["audit", "--bench", "longmemeval_s"])
└─ cli.audit(data_dir)
   ├─ data.verify_sha256(...)                               pinned dataset hash (B-001)
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
data/stores/memories/<question_id>/                                                     extracted stores (D-027)
data/res_other/preprocess_chunks/main_exp_bench/longmemeval_s/buffer/longmemeval_s_corpus.json   authors' chunks (D-028)
```

Extract the stores with `tarfile.open(path).extractall("data/stores", filter="data")`. Large files need `us.aws.cdn.hf.co` allowed (B-004).

Download with the `hf` CLI. The dataset file is named `longmemeval_s` on the Hub, with no extension:

```bash
hf download xiaowu0162/longmemeval longmemeval_s --repo-type dataset --revision 2ec2a557f339b6c0369619b1ed5793734cc87533 --local-dir data/longmemeval && mv data/longmemeval/longmemeval_s data/longmemeval/longmemeval_s.json
hf download hexuan21/MemoryTax-Res-Qwen3-5-4B --repo-type dataset --revision 987bfe5fb8317900abbad3a3260672299db1229c --include "res_simplemem_longmemeval_s_4b/*" "naive_baselines/res_rag_hybrid_1rd_longmemeval_s_4b/*" "naive_baselines/res_rag_hybrid_2rd_longmemeval_s_4b/*" --local-dir data/authors
```

## Main path: one stage

```text
cli.run(config, stage, "i/n")
└─ pipeline.run_stage(cfg, stage, i, n)        cfg from pipeline.load_config (validated)
   ├─ data.verify_sha256 + data.load_questions
   ├─ pipeline.select_questions(subset, limit) then pipeline.shard(i, n)
   ├─ pipeline.ensure_manifest(cfg)          run_id = hash of cfg.identity; written once
   ├─ pipeline.read_records(stage file)      finished questions; cuts a crash-truncated tail
   └─ for each pending question, at most cfg.concurrency at once:
      ├─ stage == "retrieve" -> system.build(q); system.retrieve(q)     see below
      ├─ stage == "answer"   -> answer.answer(client, ep, q, Retrieved)
      ├─ stage == "judge"    -> judge.judge(client, ep, q, Answer)
      ├─ LLMError            -> counted as failed, not written; a rerun retries it
      └─ pipeline.append_record(...)        one fsynced JSONL line per question
   then append the stage summary (counts, usage per operation) to manifest.json
```

Each answer and judge call goes through the same route:

```text
llm.Client.chat(ep, messages, op, prompt_version, max_tokens, attempt)
├─ key = sha256(kind, model, revision, prompt_version, payload, attempt)
├─ cache hit?  -> stored reply; usage still counted, plus cache_hits
└─ cache miss  -> POST with retries and backoff -> validate -> store -> count tokens
   └─ still failing -> failed_calls += 1; raise LLMError
```

`embed` and `rerank` follow the same route. Corpus embeddings skip the cache and are saved as `.npz` by `RagSystem.build` (D-032).

## Retrieval paths

**One-round RAG**

```text
systems/rag.RagSystem.retrieve(q)
├─ chunking.chunk_history(q)
├─ _matrix(q)                             own .npz from build(), or the authors' sample{i}.npz
├─ retrieval.BM25(texts).scores(question) -> top_k 10
├─ llm.embed(query prefix + question)     -> retrieval.dense_scores -> top_k 10
├─ retrieval.union(bm25, dense)
└─ llm.rerank(question, candidates) -> top 10; session_recall logged
```

**Two-round RAG**

```text
RagSystem.retrieve(q) as above
└─ llm.chat(SUFFICIENCY_PROMPT, answer endpoint)        D-030
   ├─ parse_sufficiency -> sufficient or unparsable -> stop
   └─ next_query -> the same round over chunks not yet picked -> round 1 + round 2 units
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
Answer session (T4 x2):
1. scripts/kaggle_serve.sh answer      4B on GPU 0; embedding and reranker on GPU 1
2. scripts/kaggle_serve.sh health      every endpoint OK, no_think=true; pin versions in serve.lock
3. memtax run --config configs/kaggle/rag1_dev100.yaml --stage retrieve     builds own dev indexes
4. memtax check-embeddings             MATCH -> set embeddings: authors in the *_all configs (D-005)
5. for C in no_context gold rag1 rag2:  memtax run --config configs/kaggle/${C}_dev100.yaml --stage retrieve, then --stage answer
6. the same for ${C}_all, then rag1_repeat20 (retrieve, answer)
7. Upload runs/ and runs/cache.sqlite to the private HF dataset, at least hourly and before the end
Judge session:
8. scripts/kaggle_serve.sh judge; scripts/kaggle_serve.sh health
9. memtax judge-agreement --dev100     >= 90% (D-009 fallback if not)
10. memtax run --config <each config> --stage judge
11. memtax gate 1; upload
```
