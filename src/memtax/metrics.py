"""Paired statistics over verdicts: accuracy, benefit, tax, net value, CIs and McNemar.

Purpose: the only place results are computed, so every table rebuilds from verdicts alone.
Called by: cli.audit() and cli.smoke(); later, the `metrics` stage.
Assumes: verdicts come from data.load_verdicts, so each is 0 or 1 and question_ids are unique
within a run. This module never calls a model, reads config, or touches the network.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import binomtest

from memtax.schema import Verdict

IntArray = npt.NDArray[np.int64]

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class PairedSummary:
    """Counts for a system against the reference (one-round RAG)."""

    n: int
    system_correct: int
    reference_correct: int
    benefit: int  # system right, reference wrong
    tax: int  # system wrong, reference right

    @property
    def net(self) -> int:
        return self.benefit - self.tax


def pair(system: list[Verdict], reference: list[Verdict]) -> tuple[IntArray, IntArray]:
    """Align two runs by question_id, refusing anything that would make the pairing invalid."""
    judges = {v.judge_id for v in system} | {v.judge_id for v in reference}
    if len(judges) != 1:
        # PAPER: D-020; a judge difference would show up as a fake benefit or tax.
        raise ValueError(f"refusing to pair verdicts from different judges: {sorted(judges)}")
    sys_map = {v.question_id: v.verdict for v in system}
    ref_map = {v.question_id: v.verdict for v in reference}
    if len(sys_map) != len(system) or len(ref_map) != len(reference):
        raise ValueError("duplicate question_id within a run")
    if sys_map.keys() != ref_map.keys():
        only = sorted(sys_map.keys() ^ ref_map.keys())
        raise ValueError(f"question sets differ in {len(only)} ids, e.g. {only[:5]}")
    if not sys_map:
        raise ValueError("no verdicts to pair")
    ids = sorted(sys_map)
    return (
        np.array([sys_map[q] for q in ids], dtype=np.int64),
        np.array([ref_map[q] for q in ids], dtype=np.int64),
    )


def paired_summary(system: IntArray, reference: IntArray) -> PairedSummary:
    s = PairedSummary(
        n=len(system),
        system_correct=int(system.sum()),
        reference_correct=int(reference.sum()),
        benefit=int(((system == 1) & (reference == 0)).sum()),
        tax=int(((system == 0) & (reference == 1)).sum()),
    )
    # WHY: checked on integer counts, so it holds exactly with no float rounding; a raise
    # rather than assert so `python -O` cannot skip it.
    if s.net != s.system_correct - s.reference_correct:
        raise AssertionError("net value identity failed")
    return s


def mcnemar_exact(summary: PairedSummary) -> float:
    """Two-sided exact McNemar p: a binomial test on the discordant pairs."""
    discordant = summary.benefit + summary.tax
    if discordant == 0:
        return 1.0
    return float(binomtest(summary.benefit, discordant, 0.5).pvalue)


def bootstrap_ci(
    per_question: IntArray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% percentile CI for the mean of a per-question vector, resampling questions.

    For paired statistics pass the per-question indicator: e.g. (sys - ref) for net value,
    so both systems are resampled together and the pairing is kept.
    """
    # shortcut: resamples questions; medication data must resample patients (Phase 4, D-015).
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_question), size=(resamples, len(per_question)))
    means = per_question[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def agreement(a: dict[str, int], b: dict[str, int]) -> float:
    """Share of questions where two verdict sets agree; for judge and replication checks.

    Unlike pair(), this deliberately compares different judges, so it never feeds tax or benefit.
    """
    if a.keys() != b.keys() or not a:
        raise ValueError(f"agreement needs the same non-empty question set ({len(a)} vs {len(b)})")
    return sum(a[q] == b[q] for q in a) / len(a)
