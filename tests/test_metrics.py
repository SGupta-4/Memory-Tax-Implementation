from pathlib import Path

import numpy as np
import pytest

from memtax import cli, data, metrics
from memtax.schema import Verdict

FIXTURES = Path(__file__).parent / "fixtures"
DATA = Path(__file__).parents[1] / "data"


def _verdicts(values: list[int], judge: str = "j") -> list[Verdict]:
    return [Verdict(f"q{i:03d}", v, judge) for i, v in enumerate(values)]


def _assert_identity(system: list[Verdict], reference: list[Verdict]) -> None:
    s_arr, r_arr = metrics.pair(system, reference)
    s = metrics.paired_summary(s_arr, r_arr)
    assert s.net == s.benefit - s.tax == s.system_correct - s.reference_correct


def test_identity_on_fixture() -> None:
    sys_v = data.load_verdicts(FIXTURES / "verdicts_system.jsonl", "fixture-judge")
    ref_v = data.load_verdicts(FIXTURES / "verdicts_reference.jsonl", "fixture-judge")
    _assert_identity(sys_v, ref_v)
    s = metrics.paired_summary(*metrics.pair(sys_v, ref_v))
    assert (s.n, s.benefit, s.tax, s.system_correct, s.reference_correct) == (4, 1, 1, 2, 2)


def test_published_ledger_counts_give_g0_numbers() -> None:
    # PAPER: authors' docs/static/js/memtax_data.js LEDGER["4b_lme"]["SimpleMem"] at 179e6a7:
    # benefit 41, both right 260, tax 91, n 500. This checks arithmetic, not the G0 audit.
    pairs = [(1, 0)] * 41 + [(1, 1)] * 260 + [(0, 1)] * 91 + [(0, 0)] * 108
    system = _verdicts([p[0] for p in pairs])
    reference = _verdicts([p[1] for p in pairs])
    s = metrics.paired_summary(*metrics.pair(system, reference))
    assert (s.system_correct / s.n, s.reference_correct / s.n) == (0.602, 0.702)
    assert (s.tax / s.n, s.benefit / s.n) == (0.182, 0.082)
    _assert_identity(system, reference)


@pytest.mark.skipif(
    not (DATA / cli.AUDIT_RUNS["simplemem"]).exists(), reason="authors' verdicts not downloaded"
)
def test_identity_on_authors_verdicts() -> None:
    runs = {
        k: data.load_verdicts(DATA / p, cli.AUTHORS_JUDGE_ID) for k, p in cli.AUDIT_RUNS.items()
    }
    _assert_identity(runs["simplemem"], runs["rag1"])
    _assert_identity(runs["rag2"], runs["rag1"])


def test_mcnemar_known_value() -> None:
    # 10 wins, 3 losses: p = 2 * sum(C(13, k), k <= 3) / 2**13 = 756 / 8192.
    s = metrics.PairedSummary(n=50, system_correct=30, reference_correct=23, benefit=10, tax=3)
    assert metrics.mcnemar_exact(s) == pytest.approx(756 / 8192, abs=1e-12)


def test_mcnemar_no_discordant_pairs() -> None:
    s = metrics.PairedSummary(n=5, system_correct=3, reference_correct=3, benefit=0, tax=0)
    assert metrics.mcnemar_exact(s) == 1.0


def test_refuses_mixed_judge_id() -> None:
    with pytest.raises(ValueError, match="different judges"):
        metrics.pair(_verdicts([1, 0], "authors"), _verdicts([1, 1], "ours"))


def test_refuses_different_question_sets() -> None:
    with pytest.raises(ValueError, match="question sets differ"):
        metrics.pair(_verdicts([1, 0, 1]), _verdicts([1, 1]))


def test_refuses_duplicate_question_id() -> None:
    dup = [Verdict("q1", 1, "j"), Verdict("q1", 0, "j")]
    with pytest.raises(ValueError, match="duplicate"):
        metrics.pair(dup, [Verdict("q1", 1, "j")])


def test_bootstrap_is_seeded_and_brackets_mean() -> None:
    x = np.array([1] * 60 + [0] * 40, dtype=np.int64)
    first, second = metrics.bootstrap_ci(x), metrics.bootstrap_ci(x)
    assert first == second
    assert first[0] < 0.6 < first[1]
