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

- **Status:** verified at G0 on 2026-11-02
- **Found by:** Reading the authors' `common/bench/bench_readme.md`.
- **Evidence:** It says LongMemEval is `xiaowu0162/longmemeval`, "not the cleaned version". `download.py` pins revision `2ec2a557f3`. The earlier plan named `longmemeval-cleaned`.
- **Tried:** 1. Changed the pinned input and recorded [D-002](DECISION.md).
- **What worked:** Using the original `longmemeval_s` file.
- **What didn't:** Nothing else was tried.
- **Verified by:** A Hub listing shows `longmemeval_s` at 278 MB. On 2026-11-02, `memtax audit` found that all three of the authors' runs have exactly the dataset's 500 question ids, and G0 passed.
- **Follow-up:** Done: `data.LONGMEMEVAL_S_SHA256` pins `08d8dad4…7894`, which matches the Hub's LFS record, and `memtax audit` checks it before loading.

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

- **Status:** verified on 2026-11-02; G0 passed
- **Scoped by:** `IMPLEMENTATION_PLAN.md`, Phase 0.
- **Evidence:** Gate G0.
- **Tried:**
  1. Built `schema.py`, `data.py`, `metrics.py`, and `cli.py` (`audit`, `smoke`), plus tests, CI, pre-commit, `scripts/check_repo.py`, and the vendored prompts with hashes.
  2. Downloads were refused while huggingface.co and then its file hosts were blocked (B-004) -> fixed by allowing them.
  3. Downloaded the authors' `judge_res/`, `qa_res/`, and `cost/` folders at `987bfe5f`, the pinned `longmemeval_s`, and the SimpleMem `memories.tar.gz`. Both large files match the Hub's LFS sha256.
- **What worked:** The audit ran on the real data unchanged, apart from reading `judge.jsonl` directly (D-026).
- **What didn't:** Nothing.
- **Cross-check:** the authors' project page data (`docs/static/js/memtax_data.js` at `179e6a7`) gives the same counts: benefit 41, both right 260, tax 91.
- **Verified by:** run on 2026-11-02 through `uv run`.

  G0:

  ```text
  $ uv run memtax audit --bench longmemeval_s
  simplemem=60.2 rag1=70.2 rag2=72.2 tax=18.2 benefit=8.2 n=500
  simplemem vs rag1: acc=60.2 [56.0, 64.4] benefit=8.2 [6.0, 10.6] tax=18.2 [14.8, 21.6] net=-10.0 [-14.4, -5.6] mcnemar_p=1.61e-05
  rag2 vs rag1: acc=72.2 [68.2, 76.0] benefit=6.4 [4.4, 8.6] tax=4.4 [2.6, 6.2] net=2.0 [-0.8, 4.8] mcnemar_p=0.22
  G0 PASS
  exit 0
  ```

  Universal gate:

  ```text
  $ uv run ruff check . && uv run ruff format --check .
  All checks passed!
  23 files already formatted
  $ uv run mypy src
  Success: no issues found in 5 source files
  $ uv run pytest -q -m "not gpu"
  24 passed
  $ uv run python scripts/check_repo.py
  OK
  $ uv run memtax smoke --config configs/fixture.yaml
  smoke OK: 4 questions, identity holds
  $ git status --short
  (only the intended files; data/ is ignored)
  ```

  Change-specific checks: `tests/test_metrics.py` covers the identity on fixture and real data, a known McNemar value, and refusal of mixed `judge_id`. `uv run pytest tests/test_vendor_hash.py -q` gives `1 passed`, and `uv lock --check` exits 0.
- **Other Phase 0 items:** the `memories.tar.gz` format is in D-027; construction cost over 500 is in B-003.
- **Follow-up:** A human tags `phase-0-pass` on this commit; tags weren't pushed by the agent. Phase 1 starts after that.

## B-004: huggingface.co blocked in the cloud session

- **Status:** fixed on 2026-11-02, by allowing hosts in the environment settings
- **2026-11-02 update:** API calls and small files work. Large (LFS and Xet) files redirect to hosts that answer `CONNECT tunnel failed, response 403`: `cas-bridge.xethub.hf.co`, `cas-server.xethub.hf.co`, `cdn-lfs.huggingface.co`, `cdn-lfs.hf.co`, and `transfer.xethub.hf.co`. This blocks `longmemeval_s` and every `memories.tar.gz`.
- **2026-11-02, second update:** after those five hosts were allowed, they answer. But both the Xet and plain LFS routes now redirect the file data to `us.aws.cdn.hf.co` (for example `/xet-bridge-us/...` for `longmemeval_s`, which is 278,025,796 bytes), and the proxy rejects that host. Allowing `us.aws.cdn.hf.co` should unblock both files.
- **Found by:** Phase 0 download step.
- **Evidence:** `curl https://huggingface.co/api/datasets/hexuan21/MemoryTax-Res-Qwen3-5-4B` -> `CONNECT tunnel failed, response 403`. The agent proxy logs `connect_rejected` for `huggingface.co:443`. GitHub and PyPI were reachable.
- **Tried:**
  1. Checked proxy status -> a policy denial, so no retry or workaround was attempted.
- **What worked:** Building and testing everything that doesn't need Hugging Face.
- **What didn't:** Nothing else was tried.
- **Verified by:** After `us.aws.cdn.hf.co` was allowed, `hf download` fetched `longmemeval_s` (278,025,796 bytes) and `memories.tar.gz` (104,803,804 bytes). Both sha256 values match the Hub's LFS records.
- **Follow-up:** A new environment needs `huggingface.co` and `us.aws.cdn.hf.co` allowed; the Xet and LFS hosts above were allowed too. Hugging Face may send other regions to another `*.cdn.hf.co` host.
