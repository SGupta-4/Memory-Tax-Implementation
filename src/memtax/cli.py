"""Command-line entry point: `memtax audit` and `memtax smoke`.

Purpose: wires data loading to metrics and prints results; holds no research logic.
Called by: the `memtax` console script (pyproject.toml), CI, and DONE.md gate commands.
Assumes: for `audit`, the pinned dataset and the authors' result folders are under --data-dir
in the layout FLOW.md gives; `smoke` needs only the files named in its config.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from memtax import data, metrics
from memtax.schema import Question, Verdict

# PAPER: the authors' judge for all published verdicts (vendor common/judge/README.md).
AUTHORS_JUDGE_ID = "authors:Qwen3.6-27B:judge/unified-v1"

# FLOW: paths under --data-dir mirror the HF repo MemoryTax-Res-Qwen3-5-4B (FLOW.md).
# WHY: read judge.jsonl only; qa_res/ also carries question_id, so walking it invites mixups.
AUDIT_RUNS = {
    "simplemem": "authors/res_simplemem_longmemeval_s_4b/judge_res/judge.jsonl",
    "rag1": "authors/naive_baselines/res_rag_hybrid_1rd_longmemeval_s_4b/judge_res/judge.jsonl",
    "rag2": "authors/naive_baselines/res_rag_hybrid_2rd_longmemeval_s_4b/judge_res/judge.jsonl",
}
AUDIT_QUESTIONS = "longmemeval/longmemeval_s.json"

# PAPER: gate G0, IMPLEMENTATION_PLAN.md section 6. Preregistered: do not edit after results.
G0_ACCURACY = {"simplemem": 60.2, "rag1": 70.2, "rag2": 72.2}
G0_TAX, G0_BENEFIT, G0_TOLERANCE, G0_N = 18.2, 8.2, 0.1, 500

SMOKE_KEYS = {"questions", "system", "reference", "judge_id"}


def _pct(count: int, n: int) -> float:
    return round(100 * count / n, 1)


def _check_ids(name: str, verdicts: list[Verdict], questions: list[Question]) -> None:
    # PAPER: B-001; matching ids is how we know the authors used the same dataset revision.
    got, want = {v.question_id for v in verdicts}, {q.question_id for q in questions}
    if got != want:
        diff = sorted(got ^ want)
        raise SystemExit(f"{name}: question ids differ from the dataset in {len(diff)}: {diff[:5]}")


def _ci_text(per_question: metrics.IntArray) -> str:
    lo, hi = metrics.bootstrap_ci(per_question)
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def audit(data_dir: Path) -> int:
    questions = data.load_questions(data_dir / AUDIT_QUESTIONS)
    runs = {k: data.load_verdicts(data_dir / p, AUTHORS_JUDGE_ID) for k, p in AUDIT_RUNS.items()}
    for name, verdicts in runs.items():
        _check_ids(name, verdicts, questions)

    ref = runs["rag1"]
    acc = {k: _pct(sum(v.verdict for v in vs), len(vs)) for k, vs in runs.items()}
    sm_sys, sm_ref = metrics.pair(runs["simplemem"], ref)
    sm = metrics.paired_summary(sm_sys, sm_ref)
    tax, benefit = _pct(sm.tax, sm.n), _pct(sm.benefit, sm.n)
    print(
        f"simplemem={acc['simplemem']} rag1={acc['rag1']} rag2={acc['rag2']} "
        f"tax={tax} benefit={benefit} n={sm.n}"
    )

    for name in ("simplemem", "rag2"):
        s_arr, r_arr = metrics.pair(runs[name], ref)
        s = metrics.paired_summary(s_arr, r_arr)
        print(
            f"{name} vs rag1: acc={_pct(s.system_correct, s.n)} {_ci_text(s_arr)} "
            f"benefit={_pct(s.benefit, s.n)} {_ci_text((s_arr > r_arr).astype(np.int64))} "
            f"tax={_pct(s.tax, s.n)} {_ci_text((s_arr < r_arr).astype(np.int64))} "
            f"net={_pct(s.net, s.n)} {_ci_text(s_arr - r_arr)} "
            f"mcnemar_p={metrics.mcnemar_exact(s):.3g}"
        )

    failures = [f"{k}={acc[k]} (want {w})" for k, w in G0_ACCURACY.items() if acc[k] != w]
    if abs(tax - G0_TAX) > G0_TOLERANCE + 1e-9:
        failures.append(f"tax={tax} (want {G0_TAX}±{G0_TOLERANCE})")
    if abs(benefit - G0_BENEFIT) > G0_TOLERANCE + 1e-9:
        failures.append(f"benefit={benefit} (want {G0_BENEFIT}±{G0_TOLERANCE})")
    if sm.n != G0_N:
        failures.append(f"n={sm.n} (want {G0_N})")
    print("G0 PASS" if not failures else "G0 FAIL: " + "; ".join(failures))
    return 0 if not failures else 1


def smoke(config_path: Path) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not SMOKE_KEYS <= config.keys():
        raise SystemExit(f"{config_path}: config needs keys {sorted(SMOKE_KEYS)}")
    # ASSUMES: paths in the config are relative to the repo root, where CI runs.
    questions = data.load_questions(Path(config["questions"]))
    system = data.load_verdicts(Path(config["system"]), str(config["judge_id"]))
    reference = data.load_verdicts(Path(config["reference"]), str(config["judge_id"]))
    _check_ids("system", system, questions)
    _check_ids("reference", reference, questions)
    summary = metrics.paired_summary(*metrics.pair(system, reference))
    print(f"smoke OK: {summary.n} questions, identity holds")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memtax")
    sub = parser.add_subparsers(dest="command", required=True)
    p_audit = sub.add_parser("audit", help="G0: statistics on the authors' verdicts")
    p_audit.add_argument("--bench", required=True, choices=["longmemeval_s"])
    p_audit.add_argument("--data-dir", type=Path, default=Path("data"))
    p_smoke = sub.add_parser("smoke", help="fixture run with no network or GPU")
    p_smoke.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "audit":
        return audit(args.data_dir)
    return smoke(args.config)


if __name__ == "__main__":
    sys.exit(main())
