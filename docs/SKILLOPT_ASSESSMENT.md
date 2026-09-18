# What was taken from SkillOpt, and what was refused

[microsoft/SkillOpt](https://github.com/microsoft/SkillOpt) (MIT) treats an
agent's skill document as trainable state and optimises it with a
validation-gated loop, leaving model weights alone. This note records which
parts of it were worth adopting here, which were not, and the evidence for
each call. Read at commit `79124b37e9a6371e13b753f8bcd7adb1e493ade1`.

The bar applied: adopt a piece only where it is measurably better than what
this repository already does, not where it is merely more elaborate.

## Taken: the paired-comparison instruments in `skillopt_sleep/evalkit.py`

`evalkit.py` is the strongest code in that repository. It refuses comparisons
whose task-id sets differ, pairs by task id, and reports McNemar's exact test
plus a percentile bootstrap — in pure stdlib, with explicit workload ceilings
and a leakage flag when the validation split is not disjoint from train.

Two functions were ported into `selfplay_lab/stats/paired.py`:

| Ported | Source | Why it is better than what was here |
| --- | --- | --- |
| `exact_sign_test` | `exact_mcnemar_p` | This repository already counted `favouring_first` and then said nothing about it. A bare "4 of 8 seeds favoured the measured wiring" has no reading without a null distribution attached. |
| `bootstrap_delta_ci` | `bootstrap_delta_ci` | Every interval here was Student-t, which assumes the per-seed differences are roughly normal. At n=8 that is an assumption nothing checks. The bootstrap assumes only exchangeability, so printing both says whether the assumption was load-bearing. |

The exact-binomial tail in the original is the careful part and was kept
verbatim in structure: it starts at the largest term of the requested tail and
recurs downward, so it neither underflows `2**-n` for large n nor converts a
huge `comb(n, k)` integer to a float.

Two deliberate deviations, both recorded in the module docstring:

- SkillOpt's bootstrap requires every score in `[0, 1]` because it compares
  success rates. Leduc returns are unbounded (around +2.6 against uniform
  random), so that check is relaxed to "finite". A gate covers it.
- SkillOpt frames the discordant-pair test as McNemar's on binary task
  outcomes. On continuous paired differences the same arithmetic is the sign
  test — both are a two-sided exact binomial with p=0.5 on the non-tied pairs.
  The function is named for the use it gets here.

### What the port immediately caught

Porting it meant re-deriving the power calculation, and the existing one was
wrong. The superseded `paired_difference` carried two hand-typed quantile
tables:

```python
t_crit  = {..., 8: 2.365, ...}   # t(0.975, df=n-1)  -- correct
t_power = {..., 8: 1.397, ...}   # t(0.90,  df=n)    -- wrong quantile, wrong df
```

The minimum detectable effect is `(t_{1-alpha/2, n-1} + t_{power, n-1}) * sem`,
with **both** quantiles at `df = n - 1`. For 80% power the second term is the
0.80 quantile. The table used the 0.90 quantile one degree of freedom too high
and labelled the answer 80% power.

Effect on the published Leduc result, re-derived from the stored per-seed
differences with no training re-run:

| Comparison | Published (labelled 80%) | Corrected 80% |
| --- | --- | --- |
| `head_to_head_real_vs_rewired` | 0.8942 | **0.7751** |
| `vs_random_real_minus_rewired` | 0.7856 | **0.6809** |

The error was conservative — it claimed the design was *less* sensitive than it
was, so it overstated how much an effect could have been hiding. The conclusion
does not move: the observed head-to-head difference is −0.357, still well inside
the corrected detection floor. But a number that happens to fail safe is still
the wrong number, and hand-typed tables are the reason nobody caught it. The
replacement computes the quantile from the incomplete beta function and is
tested against printed table values.

The tables are gone. `tests/test_paired_stats.py` holds the regression gate.

### What the port added to the existing result

Re-analysing `results/leduc_connectome_null.json` with the new instruments:

| | t 95% CI | bootstrap 95% CI | seeds favouring real | sign test |
| --- | --- | --- | --- | --- |
| head to head | [−0.919, +0.205] | [−0.808, +0.094] | 4 / 8 | p = 1.00 |
| vs-random delta | [−0.583, +0.404] | [−0.416, +0.340] | 2 / 8 | p = 0.29 |

Both intervals span zero in both rows, so the t interval was not leaning on
normality to reach its conclusion — the original report stands.

The sign test adds something the intervals could not say. Four of eight seeds
favoured the measured connectome: not "no significant difference" but *exactly*
the split the null predicts, at the largest p-value the test can return. The
measured wiring was a coin flip against its own degree-preserving rewiring.

A caution that goes with the bootstrap and is not a defect of the port: the
percentile bootstrap is anti-conservative at small n, and at n=3 it can only
ever resample three values. The smoke run at `--seeds 3` produces a bootstrap
interval that excludes zero while the t interval spans it. That disagreement is
the instrument reporting that n=3 decides nothing, which is why the runner
prints both intervals and flags when they disagree rather than picking one.

## Refused: the semantic-density term in `skillopt/evaluation/gate.py`

`select_gate_score` optionally adds a bonus to the score a candidate skill is
gated on:

```python
leading_words = ["MUST", "ALWAYS", "NEVER", "ONLY", "CRITICAL", "IMPORTANT",
                 "RESOLVE", "PREFER", "ENSURE", "STRICT", "VERIFY"]
...
score += semantic_density_weight * (leading_count / len(words))
```

The bonus is the fraction of a document's words drawn from that list, scaled by
0.05.

Scope, stated accurately: this is **opt-in and off by default**. It is reached
only through `evaluation.use_semantic_density`, which defaults to `false`, is
documented as optional in `docs/reference/config.md`, and is not enabled in any
shipped config. The nightly sleep path cannot reach it at all —
`skillopt_sleep/consolidate.py` calls `select_gate_score` positionally with four
arguments and never passes the density flags. (`gate_metric: "mixed"` in the
sleep config is an unrelated hard/soft blend.) The objection below is to the
design of an available option, not to what SkillOpt does out of the box.

Turned on, it rewards a document for how it is phrased rather than for what it makes the
agent do. A candidate that changed no instruction and only rewrote "prefer" as
"ALWAYS" would score higher. Any edit that raises accuracy by less than the
style bonus can be outvoted by one that raises nothing. The weight 0.05 and the
word list are stated without a derivation.

Not adopted, and the reason generalises: a score that mixes a measured quantity
with an aesthetic one is no longer a measurement, and you cannot tell afterwards
which half moved.

## Refused: the bare improvement gate

`evaluate_gate` accepts a candidate when `cand_score > current_score`. The score
comes from a single scored pass over the selection set — `trainer.py` computes
`cand_hard, cand_soft = compute_score(sel_results)` once and caches it by
candidate hash — so there is no repeated measurement and no noise model. When the true effect is zero, sampling noise clears that threshold about
half the time, and accepted noise becomes the baseline the next step must beat.

This repository measured the minimum detectable effect *before* spending
compute and reported that the Leduc null could not resolve anything smaller
than 0.78. Replacing that with a single-batch `>` would be a downgrade. The
honest version of the same idea is the one already in use: pair by seed, state
what the design can detect, and report the interval rather than the verdict.

## Not assessed

The trainer loop, the reflection/aggregation gradient path, the multi-backend
layer and the six benchmark environments were read only far enough to locate
the gate and the evaluation kit. Nothing is claimed about them here. The
accuracy figures in SkillOpt's README were not reproduced and are not evidence
for or against anything in this note.

## Licence

SkillOpt is MIT, Copyright (c) 2026 Microsoft Corporation. This repository is
MIT. `selfplay_lab/stats/paired.py` carries the attribution and the list of
deviations in its docstring.
