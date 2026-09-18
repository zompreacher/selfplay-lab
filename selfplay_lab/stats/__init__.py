"""Measurement instruments shared by the experiments.

Kept separate from the experiments themselves so that a claim about an
experiment's sensitivity can be tested without running the experiment.
"""

from .paired import (
    bootstrap_delta_ci,
    exact_sign_test,
    paired_difference,
    student_t_ppf,
)

__all__ = [
    "bootstrap_delta_ci",
    "exact_sign_test",
    "paired_difference",
    "student_t_ppf",
]
