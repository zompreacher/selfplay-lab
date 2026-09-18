"""Gates on the measurement instrument itself.

The connectome result in `results/` is a null: the measured wiring did not beat
its own rewiring.  A null is only worth reporting if the instrument that failed
to find an effect is known to work, so these tests check the instrument against
values that can be derived by hand or read out of a printed table, not against
whatever the code currently returns.

The `min_detectable_effect` tests exist because the first version of this
calculation was wrong in a way no test would have caught: it used the right
quantile for the interval and the wrong one for the power term.
"""

import math

import pytest

from selfplay_lab.stats import (
    bootstrap_delta_ci,
    exact_sign_test,
    paired_difference,
    student_t_ppf,
)
from selfplay_lab.stats.paired import ALPHA, POWER, PairedStatsError, student_t_cdf


# --- Student's t ---------------------------------------------------------------

# Two-sided 95% critical values, i.e. t(0.975, df), as printed in any standard
# table.  These are the numbers the interval is built from.
T_975 = {1: 12.706, 2: 4.303, 4: 2.776, 6: 2.447, 7: 2.365, 9: 2.262, 30: 2.042}


@pytest.mark.parametrize("df,expected", sorted(T_975.items()))
def test_t_ppf_matches_the_printed_table(df, expected):
    assert student_t_ppf(0.975, df) == pytest.approx(expected, abs=5e-4)


def test_t_ppf_approaches_the_normal_for_large_df():
    # z_0.975 = 1.95996, z_0.80 = 0.84162.
    assert student_t_ppf(0.975, 10_000_000) == pytest.approx(1.95996, abs=1e-3)
    assert student_t_ppf(POWER, 10_000_000) == pytest.approx(0.84162, abs=1e-3)


def test_t_cdf_and_ppf_are_inverses():
    for df in (1, 3, 7, 40):
        for p in (0.01, 0.25, 0.5, 0.8, 0.975, 0.999):
            assert student_t_cdf(student_t_ppf(p, df), df) == pytest.approx(p, abs=1e-6)


def test_t_distribution_is_symmetric():
    for df in (1, 7, 40):
        assert student_t_ppf(0.3, df) == pytest.approx(-student_t_ppf(0.7, df), abs=1e-9)


def test_t_ppf_rejects_impossible_arguments():
    for bad_p in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(PairedStatsError):
            student_t_ppf(bad_p, 7)
    with pytest.raises(PairedStatsError):
        student_t_ppf(0.5, 0)


# --- exact sign test -----------------------------------------------------------

# Two-sided exact binomial at p=0.5, computed by hand from C(8, k) / 2**8.
# Row sums: C(8,.) = 1, 8, 28, 56, 70, 56, 28, 8, 1 over 256.
SIGN_CASES = [
    (8, 0, 2 * 1 / 256),               # unanimous
    (7, 1, 2 * (1 + 8) / 256),
    (6, 2, 2 * (1 + 8 + 28) / 256),
    (5, 3, 2 * (1 + 8 + 28 + 56) / 256),
    (4, 4, 1.0),                        # exactly what the null predicts
    # 10 vs 2 out of 12: 2 * (1 + 12 + 66) / 4096.
    (10, 2, 2 * (1 + 12 + 66) / 4096),
]


@pytest.mark.parametrize("b,c,expected", SIGN_CASES)
def test_sign_test_matches_the_hand_computed_binomial(b, c, expected):
    assert exact_sign_test(b, c)["p_exact"] == pytest.approx(min(1.0, expected), abs=5e-5)


@pytest.mark.parametrize("b,c,_", SIGN_CASES)
def test_sign_test_is_symmetric_in_its_arguments(b, c, _):
    assert exact_sign_test(b, c)["p_exact"] == exact_sign_test(c, b)["p_exact"]


def test_sign_test_with_no_untied_pairs_is_not_a_result():
    out = exact_sign_test(0, 0)
    assert out["n_nonzero"] == 0
    assert out["p_exact"] == 1.0
    assert out["significant"] is False


def test_sign_test_significance_uses_the_declared_alpha():
    # 8-0 clears 0.05; 7-1 does not.  If ALPHA moves, this moves with it.
    assert exact_sign_test(8, 0)["p_exact"] < ALPHA
    assert exact_sign_test(7, 1)["p_exact"] > ALPHA
    assert exact_sign_test(8, 0)["significant"] is True
    assert exact_sign_test(7, 1)["significant"] is False


def test_sign_test_does_not_underflow_at_large_n():
    # The naive 2**-n form loses all precision here; the recurrence must not.
    p = exact_sign_test(600, 400)["p_exact"]
    assert 0.0 < p < 1e-6


def test_sign_test_rejects_non_counts():
    for bad in (-1, 1.5, True, "3"):
        with pytest.raises(PairedStatsError):
            exact_sign_test(bad, 2)


# --- bootstrap -----------------------------------------------------------------

def test_bootstrap_is_reproducible_under_a_seed():
    diffs = [0.4, -1.2, 0.3, 0.9, -0.1, 2.0, -0.7, 0.2]
    assert bootstrap_delta_ci(diffs, seed=7) == bootstrap_delta_ci(diffs, seed=7)
    assert bootstrap_delta_ci(diffs, seed=7) != bootstrap_delta_ci(diffs, seed=8)


def test_bootstrap_interval_brackets_the_sample_mean():
    diffs = [0.4, -1.2, 0.3, 0.9, -0.1, 2.0, -0.7, 0.2]
    out = bootstrap_delta_ci(diffs, n_boot=4000, seed=3)
    lo, hi = out["ci95"]
    assert lo <= sum(diffs) / len(diffs) <= hi


def test_bootstrap_of_a_constant_sample_has_zero_width():
    out = bootstrap_delta_ci([1.5] * 6, n_boot=200, seed=1)
    assert out["ci95"] == [1.5, 1.5]


def test_bootstrap_accepts_unbounded_returns():
    # The SkillOpt original restricts scores to [0, 1] because it compares
    # success rates.  Leduc returns are not in [0, 1] and must not be rejected.
    out = bootstrap_delta_ci([2.6, -3.1, 0.0, 4.4], n_boot=200, seed=1)
    assert math.isfinite(out["mean"])


def test_bootstrap_refuses_an_impossible_workload():
    with pytest.raises(PairedStatsError):
        bootstrap_delta_ci([1.0, 2.0], n_boot=10**9)


def test_bootstrap_rejects_non_finite_and_empty_samples():
    with pytest.raises(PairedStatsError):
        bootstrap_delta_ci([])
    with pytest.raises(PairedStatsError):
        bootstrap_delta_ci([1.0, float("nan")])
    with pytest.raises(PairedStatsError):
        bootstrap_delta_ci([1.0, float("inf")])


# --- the paired report ---------------------------------------------------------

def test_minimum_detectable_effect_uses_n_minus_one_degrees_of_freedom():
    """The regression gate for the bug this module was written to fix.

    MDE = (t(1-alpha/2, n-1) + t(POWER, n-1)) * sem.  The superseded version
    used t(0.90, n) for the second term -- the wrong quantile at the wrong df --
    and labelled the answer 80% power.
    """
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    b = [0.5, 2.5, 2.0, 4.5, 4.0, 6.5, 6.0, 8.5]
    out = paired_difference(a, b)
    n = len(a)
    sem = out["sem"]
    expected = (student_t_ppf(1 - ALPHA / 2, n - 1) + student_t_ppf(POWER, n - 1)) * sem
    assert out["min_detectable_effect_80pct"] == pytest.approx(expected, abs=1e-3)

    wrong = (student_t_ppf(1 - ALPHA / 2, n - 1) + student_t_ppf(0.90, n)) * sem
    assert out["min_detectable_effect_80pct"] != pytest.approx(wrong, abs=1e-4)


def test_minimum_detectable_effect_is_the_effect_the_design_would_detect():
    """An effect at the MDE should land right at the edge of significance.

    Shifting every paired difference so the mean equals the MDE leaves the
    spread untouched, so the t statistic must land at the critical value.
    """
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    b = [0.5, 2.5, 2.0, 4.5, 4.0, 6.5, 6.0, 8.5]
    out = paired_difference(a, b)
    shift = out["min_detectable_effect_80pct"] - out["mean"]
    shifted = paired_difference([x + shift for x in a], b)
    # t at the MDE = t_crit + t_power, by construction of the formula.
    n = len(a)
    assert shifted["t"] == pytest.approx(
        student_t_ppf(1 - ALPHA / 2, n - 1) + student_t_ppf(POWER, n - 1), abs=5e-3
    )


def test_paired_report_counts_both_directions_and_ties():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.0, 3.0, 3.0, 9.0]  # +1, -1, tie, -5
    out = paired_difference(a, b)
    assert out["favouring_first"] == 1
    assert out["favouring_second"] == 2
    assert out["sign_test"]["n_nonzero"] == 3


def test_paired_report_requires_matched_arms():
    with pytest.raises(PairedStatsError):
        paired_difference([1.0, 2.0], [1.0])
    with pytest.raises(PairedStatsError):
        paired_difference([], [])


def test_paired_report_carries_both_intervals():
    a = [2.0, 3.5, 1.0, 4.0, 2.5, 3.0, 1.5, 2.2]
    b = [1.0, 3.0, 1.5, 3.0, 2.0, 2.5, 1.0, 2.0]
    out = paired_difference(a, b)
    assert len(out["ci95"]) == 2
    assert len(out["bootstrap"]["ci95"]) == 2
    # Both must be reported; the point is being able to compare them.
    assert out["ci95"][0] < out["ci95"][1]
    assert out["bootstrap"]["ci95"][0] <= out["bootstrap"]["ci95"][1]


def test_sign_test_never_reports_a_p_value_of_exactly_zero():
    """A p-value of 0.0 is a certainty no finite test can deliver.

    Fixed-decimal rounding produced exactly that for large unanimous samples,
    which is why the report rounds to significant figures instead.
    """
    for b, c in [(600, 400), (5000, 4000), (200, 0)]:
        assert exact_sign_test(b, c)["p_exact"] > 0.0
