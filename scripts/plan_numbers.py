"""Reproduces every power and GPU-budget number quoted in IMPLEMENTATION_PLAN.md.

Run: python scripts/plan_numbers.py   (needs scipy). Re-run after Phase 0 and Phase 2 with
measured values; they replace the ASSUMED constants and the plan is updated if a gate changes.
"""

from statistics import mean

from scipy.stats import binom, binomtest


def mcnemar_power(n, wins_a, wins_b, alpha=0.05):
    """Exact-McNemar power. wins_a and wins_b are the discordant rates of the two systems."""
    rate, share = wins_a + wins_b, wins_a / (wins_a + wins_b)
    total = 0.0
    for d in range(1, n + 1):
        p_d = binom.pmf(d, n, rate)
        if p_d < 1e-10:
            continue
        total += p_d * sum(
            binom.pmf(k, d, share) for k in range(d + 1) if binomtest(k, d, 0.5).pvalue < alpha
        )
    return total


# Tax 18.2 from Fig. 13 (SimpleMem, Q4B, LongMemEval-s; PDF text order matched via Table 4's 27B
# value of 12.0). Benefit = net + tax = -10.0 + 18.2. Phase 0 confirms both from verdict files.
SCENARIOS = {
    "H1 SimpleMem vs RAG, 4B (tax .182, benefit .082)": (0.182, 0.082),
    "H2 src-aug gain 6 pts (ASSUMED wins .09, losses .03)": (0.09, 0.03),
    "H2 src-aug gain 3 pts (ASSUMED wins .06, losses .03)": (0.06, 0.03),
}
for name, (a, b) in SCENARIOS.items():
    print(name)
    print("   " + "  ".join(f"n={n}:{mcnemar_power(n, a, b):.2f}" for n in (100, 300, 400, 500)))

# Means over all 500 authors' cost files for SimpleMem, Qwen3.5-4B, LongMemEval-s
# (construction, MemoryTax-Res-Qwen3-5-4B at 987bfe5f; B-003). One-element lists keep mean() below.
COMPLETION = [242785.31]
PROMPT = [213164.99]
PARALLEL_S = [843.48]
SERIAL_S = [3373.91]
print(
    f"\nBuild per history: {mean(COMPLETION):,.0f} completion, {mean(PROMPT):,.0f} prompt tokens,"
    f" {mean(PARALLEL_S):.0f} s on authors' GPU (parallel), {mean(SERIAL_S):.0f} s serial"
)
print(
    f"Build 100 histories on authors' GPU: {100 * mean(PARALLEL_S) / 3600:.1f} h;"
    f" 500: {500 * mean(PARALLEL_S) / 3600:.0f} h"
)

SLOWDOWN = 3  # ASSUMED T4 vs authors' H20; replace with the rate measured in Phase 1.
RAG_INDEX_S, RAG_RETRIEVE_S, RAG_ANSWER_S = 4.1, 1.2, 0.4  # one RAG cost file
SRCAUG_S = 3.0  # ASSUMED rerank + 4.7k-token answer
MED_SHARE = 25 / 160  # 25k vs about 160k corpus tokens
hours = {
    "Index 500 histories": 500 * RAG_INDEX_S,
    "RAG one round, 500": 500 * (RAG_RETRIEVE_S + RAG_ANSWER_S),
    "RAG two rounds, 500": 500 * 2 * (RAG_RETRIEVE_S + RAG_ANSWER_S),
    "SimpleMem replay, 500": 500 * SRCAUG_S,
    "Source augmentation, 500": 500 * SRCAUG_S,
    "SimpleMem own build, 10": 10 * mean(PARALLEL_S),
    "Medication build, 30 patients": 30 * mean(PARALLEL_S) * MED_SHARE,
    "Medication answering, 360 q (3 systems)": 3 * 360 * SRCAUG_S,
}
print(f"\nGPU hours at slowdown x{SLOWDOWN} (before judge and smoke runs):")
total = 0.0
for name, sec in hours.items():
    h = sec * SLOWDOWN / 3600
    total += h
    print(f"  {name:42s}{h:6.1f}")
print(f"  {'Judge and smoke runs (ASSUMED)':42s}{3.0:6.1f}")
print(f"  {'TOTAL':42s}{total + 3:6.1f}")
