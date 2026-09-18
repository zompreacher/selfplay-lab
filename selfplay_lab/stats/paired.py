"""Paired-sample statistics for arms that share seeds.

Every comparison in this repository is paired: the measured connectome and its
degree-preserving rewiring are trained on the same seeds, so the seed is a
block and the unit of analysis is the per-seed difference.  This module is the
instrument for those differences.

It reports three things about the same paired sample, deliberately:

* a Student-t interval, which is sensitive but assumes the per-seed
  differences are roughly normal -- an assumption, not a measurement, at n=8;
* a percentile bootstrap interval, which assumes only exchangeability;
* an exact sign test on how many seeds favoured each arm, which assumes only
  that a tie-free difference is equally likely to fall either way under the
  null.

Two intervals that agree mean the normality assumption was not load-bearing.
Two that disagree mean the t interval was doing work the data does not
support, and the distribution-free one is the honest number.  Reporting only
the t interval hides which case you are in.

Pure stdlib.  `scipy` would supply `t.ppf`, `binomtest` and a bootstrap, but
the experiment runners already carry torch and OpenSpiel, and a statistics
dependency that only four call sites use is not worth the install surface.

Provenance
----------
`exact_sign_test` and `bootstrap_delta_ci` are ports of `exact_mcnemar_p` and
`bootstrap_delta_ci` from `skillopt_sleep/evalkit.py` in microsoft/SkillOpt
(MIT, Copyright (c) 2026 Microsoft Corporation), which is the part of that
project worth taking: a careful, dependency-free exact binomial tail and a
paired bootstrap, both with explicit workload limits.

Two deliberate deviations from the source:

* SkillOpt's bootstrap requires every score in [0, 1] because it compares
  success rates.  The returns compared here are unbounded game returns
  (Leduc pays out around +2.6), so that check is relaxed to "finite".
* SkillOpt frames the discordant-pair test as McNemar's on binary task
  outcomes.  The arithmetic is identical for the sign test on continuous
  paired differences -- both are a two-sided exact binomial with p=0.5 on the
  non-tied pairs -- so the function is named for the use it gets here.

What was NOT taken, and why, is recorded in docs/SKILLOPT_ASSESSMENT.md.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Iterable, Sequence

# Ceilings copied in spirit from the source: a bootstrap that silently takes an
# hour is a worse failure than one that refuses.
MAX_BOOTSTRAPS = 1_000_000
MAX_BOOTSTRAP_DRAWS = 50_000_000

# Conventional two-sided level and power for the reported interval and the
# minimum detectable effect.  Named rather than inlined so the numbers in the
# results files can be traced back to a choice rather than to a habit.
ALPHA = 0.05
POWER = 0.80


class PairedStatsError(ValueError):
    """A contract failure in the sample handed to these functions."""


# --- distributions -------------------------------------------------------------


def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-16) -> float:
    """Continued-fraction expansion for the incomplete beta function."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    # Use whichever of I_x(a,b) / 1 - I_{1-x}(b,a) has the faster-converging
    # continued fraction.
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: int) -> float:
    """P(T <= t) for Student's t with `df` degrees of freedom."""
    if df < 1:
        raise PairedStatsError("degrees of freedom must be at least 1")
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + float(t) * float(t)))
    return 1.0 - tail if t > 0 else tail


def student_t_ppf(p: float, df: int) -> float:
    """Inverse CDF of Student's t, by bisection on the CDF.

    Computed rather than read from a hand-typed table.  The table this replaces
    held the right quantiles for the interval and the wrong ones for the power
    term -- see docs/SKILLOPT_ASSESSMENT.md.  A table is a number nobody can
    check without leaving the file.
    """
    if not 0.0 < p < 1.0:
        raise PairedStatsError("p must be strictly between 0 and 1")
    if df < 1:
        raise PairedStatsError("degrees of freedom must be at least 1")
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- exact sign test -----------------------------------------------------------


def _binom_pmf_half(k: int, n: int) -> float:
    """Binomial pmf at p=0.5, via log-gamma so C(n, k) never overflows a float."""
    if k < 0 or k > n:
        return 0.0
    return math.exp(
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        - n * math.log(2.0)
    )


def exact_sign_test(n_favouring_first: int, n_favouring_second: int) -> dict:
    """Two-sided exact binomial test on the non-tied pairs, p=0.5.

    This is the sign test on paired differences, and simultaneously the exact
    form of McNemar's test on paired binary outcomes -- the same arithmetic
    under two names.  It answers the question the bare count in a results file
    leaves open: 4 of 8 seeds favouring one arm is what the null predicts, and
    8 of 8 is not, but nothing in the count itself says where the line is.

    It assumes only that a non-tied difference is equally likely to fall either
    way under the null.  No normality, no equal variances, no interval.
    """
    b = _validate_count("n_favouring_first", n_favouring_first)
    c = _validate_count("n_favouring_second", n_favouring_second)
    n = b + c
    if n == 0:
        # Every pair tied.  No evidence either way, and no test is defined.
        return {"n_nonzero": 0, "p_exact": 1.0, "significant": False}

    k = min(b, c)

    def lower_tail_terms() -> Iterable[float]:
        # Start at the largest term of the requested tail and recur downward:
        # pmf(i-1)/pmf(i) = i / (n - i + 1).  Starting from pmf(0) = 2**-n
        # underflows for large n, and comb(n, k) * 0.5**n converts a huge int
        # to float.  This does neither.
        term = _binom_pmf_half(k, n)
        yield term
        for i in range(k, 0, -1):
            term *= i / (n - i + 1)
            yield term

    # p = 0.5 makes the distribution symmetric, so doubling one tail is exact.
    p_exact = min(1.0, 2.0 * math.fsum(lower_tail_terms()))
    return {
        "n_nonzero": n,
        "p_exact": _significant_figures(p_exact),
        "significant": p_exact < ALPHA,
    }


def _significant_figures(x: float, digits: int = 4) -> float:
    """Round to significant figures, not decimal places.

    A p-value rounded to four decimals reports 2.7e-10 as 0.0, and a p-value is
    never exactly zero.  Rounding to significant figures keeps a small p small
    instead of inventing a certainty the test cannot deliver.
    """
    if x == 0.0 or not math.isfinite(x):
        return x
    return float(f"{x:.{digits}g}")


def _validate_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PairedStatsError(f"{name} must be a non-negative integer")
    return value


# --- bootstrap -----------------------------------------------------------------


def bootstrap_delta_ci(
    diffs: Sequence[float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = ALPHA,
) -> dict:
    """Percentile bootstrap interval on the mean paired difference.

    Resamples the per-seed differences with replacement, so the pairing is
    preserved by construction: a seed is drawn as a unit or not at all.

    Assumes exchangeability of the seeds and nothing about their distribution,
    which is the point -- it is the check on the t interval sitting beside it.
    At n=8 it is coarse (the resampling can only ever see the 8 observed
    values), so a bootstrap interval that is much wider than the t interval
    means the t interval was extrapolating.
    """
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or not 1 <= n_boot <= MAX_BOOTSTRAPS:
        raise PairedStatsError(f"n_boot must be an integer in 1..{MAX_BOOTSTRAPS}")
    if not 0.0 < alpha < 1.0:
        raise PairedStatsError("alpha must be strictly between 0 and 1")
    values = list(diffs)
    n = len(values)
    if n == 0:
        raise PairedStatsError("bootstrap requires a non-empty paired sample")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in values):
        raise PairedStatsError("differences must be numbers")
    values = [float(x) for x in values]
    if any(not math.isfinite(x) for x in values):
        raise PairedStatsError("differences must be finite")
    if n_boot > MAX_BOOTSTRAP_DRAWS // n:
        raise PairedStatsError(
            f"bootstrap workload exceeds the {MAX_BOOTSTRAP_DRAWS} paired-draw "
            "limit (n * n_boot)"
        )

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(math.fsum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_i = max(0, min(n_boot - 1, int(math.floor((alpha / 2.0) * (n_boot - 1)))))
    hi_i = max(0, min(n_boot - 1, int(math.ceil((1.0 - alpha / 2.0) * (n_boot - 1)))))
    return {
        "n_boot": n_boot,
        "seed": seed,
        "ci95": [round(means[lo_i], 4), round(means[hi_i], 4)],
        "mean": round(math.fsum(means) / n_boot, 4),
    }


# --- the report ----------------------------------------------------------------


def paired_difference(a_values, b_values, *, bootstrap_seed: int = 42):
    """Paired difference between two arms that share seeds.

    Reported rather than a bare p-value: with small n the interval says what
    the experiment could and could not have detected, which is the part that
    matters when the answer comes back null.  The distribution-free companions
    say whether the interval can be believed.
    """
    if len(a_values) != len(b_values):
        raise PairedStatsError("paired arms must have the same number of seeds")
    diffs = [a - b for a, b in zip(a_values, b_values)]
    n = len(diffs)
    if n == 0:
        raise PairedStatsError("paired arms must be non-empty")
    mean = statistics.mean(diffs)
    if n < 2:
        return {"mean": round(mean, 4), "n": n}

    sd = statistics.stdev(diffs)
    sem = sd / math.sqrt(n)
    df = n - 1
    t_crit = student_t_ppf(1.0 - ALPHA / 2.0, df)
    # Minimum detectable effect: the normal-approximation form of the paired-t
    # power calculation, ncp ~= t_{1-alpha/2, df} + t_{power, df}.  BOTH
    # quantiles take df = n - 1.  The table this replaced used the 0.90
    # quantile at df = n for the second term while labelling the result 80%
    # power, which reported a 90%-power figure one df too high.
    t_power = student_t_ppf(POWER, df)

    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)

    return {
        "mean": round(mean, 4),
        "std": round(sd, 4),
        "sem": round(sem, 4),
        "ci95": [round(mean - t_crit * sem, 4), round(mean + t_crit * sem, 4)],
        "t": round(mean / sem, 3) if sem else None,
        "n": n,
        "favouring_first": n_pos,
        "favouring_second": n_neg,
        "sign_test": exact_sign_test(n_pos, n_neg),
        "bootstrap": bootstrap_delta_ci(diffs, seed=bootstrap_seed, alpha=ALPHA),
        "min_detectable_effect_80pct": round((t_crit + t_power) * sem, 4),
        "per_seed": [round(d, 4) for d in diffs],
    }
