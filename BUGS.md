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

- **Status:** verified at plan level; re-measure in Phase 2
- **Found by:** Reading authors' cost files.
- **Evidence:** 9 sampled histories average 250,747 completion and 215,682 prompt tokens, and 993 seconds on the authors' GPUs. The earlier formula divided total tokens by an assumed 1,500 tokens per second, which gave about 7 hours for 100 histories. At the sampled time, 100 histories take 27.6 hours before any T4 slowdown.
- **Tried:**
  1. Cut to 30 builds -> still about 25 T4-hours, and too few for H1 or H2.
- **What worked:** [D-004](DECISION.md): 10 own builds for fidelity, and authors' stores for statistics.
- **What didn't:** Scaling down builds alone.
- **Verified by:** Running `python scripts/plan_numbers.py`.
- **Follow-up:** The sample of 9 isn't random: I picked them from the start of the alphabetical file list. Recalculate over all 500 in Phase 0.

## F-001: Phase 0 audit

- **Status:** scoped
- **Scoped by:** `IMPLEMENTATION_PLAN.md`, Phase 0.
- **Evidence:** Gate G0.
- **Tried:** Not started.
- **Verified by:** To fill in with `DONE.md` output.
