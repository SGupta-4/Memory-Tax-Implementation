"""Command-line entry point: audit, check-chunks, run, and smoke.

Purpose: wires data loading, the pipeline, and metrics to commands; holds no research logic.
Called by: the `memtax` console script (pyproject.toml), CI, and DONE.md gate commands.
Assumes: downloads sit under --data-dir in the layout FLOW.md gives; `smoke` runs from a
checkout, because it loads tests/fake_server.py by path (it is not part of the package).
"""

import argparse
import asyncio
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from memtax import data, gates, metrics, pipeline, vendor
from memtax.schema import Question, Verdict

# PAPER: the authors' judge for all published verdicts (vendor common/judge/README.md).
AUTHORS_JUDGE_ID = "authors:Qwen3.6-27B:judge/unified-v1"

# FLOW: paths under --data-dir mirror the HF repo MemoryTax-Res-Qwen3-5-4B (FLOW.md).
# WHY: read judge.jsonl only; qa_res/ also carries question_id, so walking it invites mixups.
AUDIT_RUNS = {
    "simplemem": "authors/res_simplemem_longmemeval_s_4b/judge_res/judge.jsonl",
    "rag1": gates.AUTHORS_RAG1_VERDICTS,
    "rag2": "authors/naive_baselines/res_rag_hybrid_2rd_longmemeval_s_4b/judge_res/judge.jsonl",
}
AUDIT_QUESTIONS = "longmemeval/longmemeval_s.json"

# PAPER: gate G0, IMPLEMENTATION_PLAN.md section 6. Preregistered: do not edit after results.
G0_ACCURACY = {"simplemem": 60.2, "rag1": 70.2, "rag2": 72.2}
G0_TAX, G0_BENEFIT, G0_TOLERANCE, G0_N = 18.2, 8.2, 0.1, 500


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
    data.verify_sha256(data_dir / AUDIT_QUESTIONS, data.LONGMEMEVAL_S_SHA256)
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
    """Run every stage for one- and two-round RAG on the fixture, against the fake server."""
    fake_path = vendor.ROOT.parents[1] / "tests" / "fake_server.py"
    spec = importlib.util.spec_from_file_location("memtax_fake_server", fake_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{fake_path}: smoke needs a repo checkout")
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp, fake.FakeServer() as server:
        for ep in raw["endpoints"].values():
            ep["base_url"] = server.base_url
        raw.update(runs_dir=f"{tmp}/runs", cache=f"{tmp}/cache.sqlite", index_dir=f"{tmp}/index")
        verdicts: dict[int, list[Verdict]] = {}
        for rounds in (1, 2):
            raw["system"]["rounds"] = rounds
            cfg = pipeline.parse_config(raw)
            for stage in pipeline.STAGE_FILES:
                s = asyncio.run(pipeline.run_stage(cfg, stage))
                if s["failed"] or s["missing_input"]:
                    raise SystemExit(f"smoke FAIL: rounds={rounds} {stage}: {s}")
            judge = str(pipeline.ensure_manifest(cfg)["judge_id"])
            verdicts[rounds] = data.load_verdicts(cfg.run_dir / "verdicts.jsonl", judge)
    summary = metrics.paired_summary(*metrics.pair(verdicts[2], verdicts[1]))
    print(f"smoke OK: {summary.n} questions, identity holds")
    return 0


def run(config_path: Path, stage: str, shard_text: str) -> int:
    index, _, count = shard_text.partition("/")
    cfg = pipeline.load_config(config_path)
    s = asyncio.run(pipeline.run_stage(cfg, stage, int(index), int(count or 1)))
    line = (
        f"{stage} run={cfg.run_id} shard={s['shard']}: {s['done']} done, "
        f"{s['already_done']} already done, {s['failed']} failed, "
        f"{s['missing_input']} missing input"
    )
    if stage == "retrieve":
        recalls = [
            r["session_recall"]
            for r in pipeline.read_records(cfg.run_dir / "retrieval.jsonl").values()
            if r["session_recall"] is not None
        ]
        if recalls:
            line += f", session recall {sum(recalls) / len(recalls):.3f}"
    print(line)
    # FLOW: failed questions are not written, so rerunning the same command retries them.
    return 1 if s["failed"] else 0


def check_chunks(data_dir: Path) -> int:
    matched, n, bad = gates.chunk_match(data_dir)
    minimum = float(gates.thresholds()["chunk_match_min"])
    print(f"chunks match: {matched}/{n} histories ({100 * matched / n:.1f}%)")
    if bad:
        print(f"first mismatches: {bad}")
    ok = matched / n >= minimum
    print(f"G1 chunks {'PASS' if ok else 'FAIL'} (need {100 * minimum:.0f}%)")
    return 0 if ok else 1


def check_embeddings(config: Path, data_dir: Path) -> int:
    r = asyncio.run(gates.check_embeddings(pipeline.load_config(config), data_dir))
    print(
        f"corpus cosine mean={r['corpus_mean']:.5f} min={r['corpus_min']:.5f}; "
        f"query cosine mean={r['query_mean']:.5f} min={r['query_min']:.5f}"
    )
    ok = min(r["corpus_mean"], r["query_mean"]) >= gates.EMBED_COSINE_MIN
    print(f"embeddings {'MATCH: reuse the authors' if ok else 'DIFFER: use own'} (D-005)")
    return 0 if ok else 1


def judge_agreement(config: Path, data_dir: Path) -> int:
    cfg = pipeline.load_config(config)
    share, judged, unjudged = asyncio.run(gates.judge_agreement(cfg, data_dir))
    minimum = float(gates.thresholds()["judge_agreement_min"])
    print(
        f"judge agreement with authors: {share:.3f} on {judged} dev questions, {unjudged} unjudged"
    )
    ok = share >= minimum and unjudged == 0
    print(f"G1 judge {'PASS' if ok else 'FAIL'} (need {minimum:.2f})")
    return 0 if ok else 1


def gate(number: int, rag1: Path, repeat: Path, data_dir: Path) -> int:
    if number != 1:
        raise SystemExit(f"gate {number} is not built yet")
    checks = gates.gate1(pipeline.load_config(rag1), pipeline.load_config(repeat), data_dir)
    for text, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {text}")
    passed = all(ok for _, ok in checks)
    print("G1 PASS" if passed else "G1 FAIL")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memtax")
    sub = parser.add_subparsers(dest="command", required=True)
    p_audit = sub.add_parser("audit", help="G0: statistics on the authors' verdicts")
    p_audit.add_argument("--bench", required=True, choices=["longmemeval_s"])
    p_audit.add_argument("--data-dir", type=Path, default=Path("data"))
    p_smoke = sub.add_parser("smoke", help="fixture run with no network or GPU")
    p_smoke.add_argument("--config", type=Path, required=True)
    p_chunks = sub.add_parser("check-chunks", help="G1: compare chunks with the authors'")
    p_chunks.add_argument("--against", required=True, choices=["authors"])
    p_chunks.add_argument("--data-dir", type=Path, default=Path("data"))
    p_run = sub.add_parser("run", help="run one stage of a config; resumable")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--stage", required=True, choices=list(pipeline.STAGE_FILES))
    p_run.add_argument("--shard", default="0/1", help="i/n: this process runs shard i of n")
    p_emb = sub.add_parser("check-embeddings", help="D-005: own dev-100 embeddings vs authors'")
    p_emb.add_argument("--config", type=Path, default=Path("configs/kaggle/rag1_dev100.yaml"))
    p_emb.add_argument("--data-dir", type=Path, default=Path("data"))
    p_judge = sub.add_parser("judge-agreement", help="G1: own judge on the authors' answers")
    p_judge.add_argument("--dev100", action="store_true", required=True)
    p_judge.add_argument("--config", type=Path, default=Path("configs/kaggle/rag1_all.yaml"))
    p_judge.add_argument("--data-dir", type=Path, default=Path("data"))
    p_gate = sub.add_parser("gate", help="check a phase gate")
    p_gate.add_argument("number", type=int)
    p_gate.add_argument("--rag1", type=Path, default=Path("configs/kaggle/rag1_all.yaml"))
    p_gate.add_argument("--repeat", type=Path, default=Path("configs/kaggle/rag1_repeat20.yaml"))
    p_gate.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    if args.command == "audit":
        return audit(args.data_dir)
    if args.command == "check-chunks":
        return check_chunks(args.data_dir)
    if args.command == "run":
        return run(args.config, args.stage, args.shard)
    if args.command == "check-embeddings":
        return check_embeddings(args.config, args.data_dir)
    if args.command == "judge-agreement":
        return judge_agreement(args.config, args.data_dir)
    if args.command == "gate":
        return gate(args.number, args.rag1, args.repeat, args.data_dir)
    return smoke(args.config)


if __name__ == "__main__":
    sys.exit(main())
