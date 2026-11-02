# Bugs and features

One entry per bug or feature, traced from start to finish. IDs are `B-NNN` for bugs and `F-NNN` for features. Newest entries go at the bottom.

**Template**

```markdown
## B-NNN or F-NNN: Title
- **Status:** scoped | in progress | verified | abandoned
- **Found or scoped by:** How it surfaced, or what requirement it serves.
- **Evidence:** Exact error, numbers, or file and line.
- **Tried:**
  1. What was tried -> what happened.
- **What worked:** The change, with commit.
- **What didn't:** Why the other attempts failed.
- **Verified by:** Commands from DONE.md, with actual output.
- **Follow-up:** Any remaining work.
```

## B-001: Plan used the wrong dataset version

- **Status:** verified at plan level; recheck at G0
- **Found by:** Reading the authors' `common/bench/bench_readme.md`.
- **Evidence:** It says LongMemEval is `xiaowu0162/longmemeval`, "not the cleaned version". `download.py` pins revision `2ec2a557f3`. The earlier plan named `longmemeval-cleaned`.
- **Tried:** 1. Changed the pinned input and recorded [D-002](DECISION.md).
- **What worked:** Using the original `longmemeval_s` file.
- **What didn't:** Nothing else was tried.
- **Verified by:** A Hub listing shows `longmemeval_s` at 278 MB. G0 checks question IDs against the authors' verdicts.
- **Follow-up:** Assert the file's hash in `data.py`.

## B-002: 100 questions couldn't reliably detect the effect

- **Status:** verified at plan level
- **Found by:** Power calculation during planning.
- **Evidence:** `scripts/plan_numbers.py` gives 0.42 power for H1 at n=100 and 0.31 for a 6-point H2. The subset's `used_for` field reads "GPT-5.6-luna (low) backbone only; Qwen backbones use all 500 questions".
- **Tried:**
  1. Keep 100 and report wide CIs -> rejected, as a null would mean little.
  2. Build more histories -> rejected by B-003.
- **What worked:** Replaying authors' stores on all 500. H1 power is 0.99; a 6-point H2 on the held-out 400 is 0.93.
- **What didn't:** Options 1 and 2 above.
- **Verified by:** Running `python scripts/plan_numbers.py`.
- **Follow-up:** Replace assumed H2 win and loss rates with dev-set values.

## B-003: Plan's SimpleMem build budget was far too low

- **Status:** verified over all 500 cost files on 2026-11-02; re-measure own builds in Phase 2
- **Found by:** Reading authors' cost files.
- **Evidence:** 9 sampled histories average 250,747 completion and 215,682 prompt tokens, and 993 seconds on the authors' GPUs. The earlier formula divided total tokens by an assumed 1,500 tokens per second, which gave about 7 hours for 100 histories. At the sampled time, 100 histories take 27.6 hours before any T4 slowdown.
- **Tried:**
  1. Cut to 30 builds -> still about 25 T4-hours, and too few for H1 or H2.
- **What worked:** [D-004](DECISION.md): 10 own builds for fidelity, and authors' stores for statistics.
- **What didn't:** Scaling down builds alone.
- **Verified by:** Running `python scripts/plan_numbers.py`.
- **Follow-up:** The sample of 9 isn't random: I picked them from the start of the alphabetical file list. Recalculate over all 500 in Phase 0.
- **2026-11-02, over all 500:** construction averages 242,785 completion and 213,165 prompt tokens, and 843 s parallel (median 868, range 287 to 1,687) on the authors' GPUs. The 9-history sample overstated time by about 15%. 100 builds take 23.4 h, not 27.6 h; 500 take 117 h, not 138 h. At the assumed 3x slowdown, 10 own builds take 7.0 T4-hours, not 8.3, and the plan total falls from about 24 to about 22 GPU hours. D-004 still holds; no gate changed. `scripts/plan_numbers.py` and plan section 5 are updated.

## F-001: Phase 0 audit

- **Status:** in progress. The authors' verdicts reproduce every G0 number, but the G0 command can't run until the dataset file downloads (B-004)
- **2026-11-02 update:** huggingface.co became reachable, but its large-file hosts are still denied. The authors' `judge_res/`, `qa_res/`, and `cost/` folders downloaded at revision `987bfe5f` (24 MB). `longmemeval_s` (278 MB) and `memories.tar.gz` (105 MB) did not. The verdict layout is recorded in D-026. On the real verdicts, through `memtax.metrics`:

  ```text
  simplemem n 500 acc 60.2 ref 70.2 benefit 41 8.2 tax 91 18.2 net CI (-14.4, -5.6) p=1.61e-05
  rag2      n 500 acc 72.2 ref 70.2 benefit 32 6.4 tax 22 4.4  net CI (-0.8, 4.8)  p=0.22
  ```

  That matches G0's targets exactly: 60.2, 70.2, 72.2, tax 18.2, benefit 8.2, n=500. All 1,500 verdicts are 0 or 1, with none null. This is **not yet G0**: `memtax audit` also checks question ids against `longmemeval_s.json` (B-001), and that check hasn't run. `test_identity_on_authors_verdicts` now runs and passes.
- **Scoped by:** `IMPLEMENTATION_PLAN.md`, Phase 0.
- **Evidence:** Gate G0.
- **Tried:**
  1. Built `schema.py`, `data.py`, `metrics.py`, and `cli.py` (`audit`, `smoke`), plus tests, CI, pre-commit, `scripts/check_repo.py`, and the vendored prompts with hashes -> the universal gate passes (below).
  2. Downloaded the pinned dataset and the authors' result folders -> refused: huggingface.co is denied by this session's network policy (B-004).
- **What worked:** Everything that doesn't need the downloads.
- **What didn't:** The G0 run, the `memories.tar.gz` inspection, and the construction-cost recount over 500 histories. None of these were run, and no G0 result is claimed.
- **Cross-check that isn't G0:** the authors' project page data (`docs/static/js/memtax_data.js` at `179e6a7`, `LEDGER["4b_lme"]["SimpleMem"]`) gives benefit 41, both right 260, tax 91, n 500. That is 60.2% SimpleMem, 70.2% one-round RAG, tax 18.2%, and benefit 8.2%, matching G0's targets. `test_published_ledger_counts_give_g0_numbers` asserts this arithmetic. It confirms the plan's figure reading, not our audit of per-question verdicts.
- **Verified by:** universal gate, run 2026-11-02 through `uv run`:

  ```text
  $ uv run ruff check . && uv run ruff format --check .
  All checks passed!
  23 files already formatted
  $ uv run mypy src
  Success: no issues found in 5 source files
  $ uv run pytest -q -m "not gpu"
  23 passed
  $ uv run python scripts/check_repo.py
  OK
  $ uv run memtax smoke --config configs/fixture.yaml
  smoke OK: 4 questions, identity holds
  $ uv run pytest tests/test_vendor_hash.py -q
  1 passed
  $ uv lock --check
  exit 0
  ```

  Rerun on 2026-11-02 after the verdict download; `test_identity_on_authors_verdicts` now runs and passes. `uv run memtax audit --bench longmemeval_s` stops with `FileNotFoundError: ... data/longmemeval/longmemeval_s.json`, as expected.
- **Follow-up:** once the large-file hosts in B-004 are reachable (steps 1 and 4 are partly or fully done):
  1. Download `longmemeval_s` and the SimpleMem `memories.tar.gz` as in `FLOW.md` "Phase 0 data layout". The verdict, QA and cost files are already in `data/authors/` at `987bfe5f`.
  2. Run `uv run memtax audit --bench longmemeval_s`; paste its output here. If the loader rejects the files, fix it under D-023.
  3. Inspect `memories.tar.gz` (extract with `filter="data"`) and record its format in `DECISION.md`.
  4. Done: construction cost recalculated over all 500 (B-003).
  5. If G0 passes, tag `phase-0-pass` and add the phase note to `REVIEW.md`.

## B-004: huggingface.co blocked in the cloud session

- **Status:** partly fixed on 2026-11-02. huggingface.co is reachable; large-file CDN hosts are still denied
- **2026-11-02 update:** API calls and small files work. Large (LFS and Xet) files redirect to hosts that answer `CONNECT tunnel failed, response 403`: `cas-bridge.xethub.hf.co`, `cas-server.xethub.hf.co`, `cdn-lfs.huggingface.co`, `cdn-lfs.hf.co`, and `transfer.xethub.hf.co`. This blocks `longmemeval_s` and every `memories.tar.gz`.
- **Found by:** Phase 0 download step.
- **Evidence:** `curl https://huggingface.co/api/datasets/hexuan21/MemoryTax-Res-Qwen3-5-4B` -> `CONNECT tunnel failed, response 403`. The agent proxy logs `connect_rejected` for `huggingface.co:443`. GitHub and PyPI were reachable.
- **Tried:**
  1. Checked proxy status -> a policy denial, so no retry or workaround was attempted.
- **What worked:** Building and testing everything that doesn't need Hugging Face.
- **What didn't:** Nothing else was tried.
- **Verified by:** Not fixed yet.
- **Follow-up:** Add `huggingface.co` (and `cdn-lfs.huggingface.co`, `cas-bridge.xethub.hf.co` for large files) to the environment's allowed domains, or run the F-001 follow-up on a machine with access.
