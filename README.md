# selfplay-lab

Two NFSP variants for imperfect-information, N-player games, extracted from a private
research lab and made game-agnostic. Runs on any [OpenSpiel](https://github.com/google-deepmind/open_spiel)
game. PyTorch, CPU.

This is the **crawl** release: the core algorithms, a minimal trainer, and the gates
that prove each variant is doing what its name says. The full lab (a Rust engine core,
a 500k-decision differential oracle, interpretability probes, a 50-experiment research
log) is built on a licensed commercial card game and stays private until its publisher
signs off on what can be released.

## What is here

| File | What it is |
|---|---|
| `selfplay_lab/nfsp_distil.py` | **DistilNFSP.** Stock NFSP trains its deployable average-policy net on a reservoir of (state, action-probs) pairs whose labels are frozen at insertion time. In a 4-player self-play run that reservoir filled at ~32k episodes, new play entered with probability falling like 1/n, and the average policy lagged the live best response by ~30 percentage points. This subclass changes ONE thing: at every SL step the target is recomputed from the current Q head (softmax over standardised legal Q values at a temperature) instead of read from storage. The standardisation is the design: it makes the policy invariant to the Q scale, so Adam and SGD runs agree on what the policy is. |
| `selfplay_lab/nfsp_search.py` | **SearchNFSP.** The AlphaZero shape mapped onto NFSP's two heads: in best-response mode the agent runs a net-guided (PUCT prior from the average net), leaf-evaluated (Q head, no rollout) ISMCTS, plays the searched action, and stores the search's normalised visit distribution as the SL label. Dirichlet root noise with alpha derived from the branching factor, not copied from Go. Every knob the arm depends on is asserted at construction so a silent fallback to stock NFSP cannot train a different experiment under this one's name. |
| `selfplay_lab/train.py` | Minimal trainer: one agent per seat, `--agent {stock,distil,search}`, checkpoint and resume, evaluation on a fixed seed base (common random numbers) so any two checkpoints are scored on the same deals. Reports win rate (fair line 1/N) and mean return (fair line 0). |
| `selfplay_lab/flybrain/` | **A measured fly connectome as a fixed reservoir.** The Janelia hemibrain central complex - 449 neurons, 63,417 EM-reconstructed connections - placed in front of the agents as a recurrent feature map. The wiring is a buffer, never a parameter: training its measured weights would delete the measurement and leave an ordinarily-initialised RNN with an unusual topology. Per-neuron gain and time constant *are* trainable and default to frozen, because an adjacency table does not measure intrinsic excitability. `nulls.py` holds the controls the whole thing stands on. |
| `selfplay_lab/stats/` | **The instrument the null results depend on.** Paired-sample statistics for arms that share seeds: a Student-t interval, a percentile bootstrap that assumes only exchangeability, and an exact sign test on the per-seed direction. All three are reported together, because an interval that disagrees with its distribution-free companion is leaning on an assumption the sample size cannot check. Pure stdlib; the t quantile is computed, not read from a table. Partly ported from microsoft/SkillOpt (MIT) - see `docs/SKILLOPT_ASSESSMENT.md`. |
| `selfplay_lab/experiments/` | `connectome_null.py` runs the measured wiring against its own degree-preserving rewiring; `reservoir_diagnostics.py` measures how much of a difference was available to find in the first place. |
| `tests/` | Gates, not smoke: affine invariance of the distil target, degenerate-spread fallback, PUCT + visit-count policy asserted on the constructed bot, search calls and net forwards counted, leaf projection matches `num_players`, all three variants train / checkpoint / resume. |

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -q                     # 11 tests, ~5 s on CPU

# 3-player Leduc poker, the three variants
python -m selfplay_lab.train --game "leduc_poker(players=3)" --agent stock  --episodes 6000
python -m selfplay_lab.train --game "leduc_poker(players=3)" --agent distil --episodes 6000
python -m selfplay_lab.train --game "leduc_poker(players=3)" --agent search --search-sims 16 --episodes 2000

# any OpenSpiel game with an information_state_tensor works; search additionally
# needs resample_from_infostate (kuhn_poker, leduc_poker, and most card games have it)
python -m selfplay_lab.train --game "kuhn_poker(players=3)" --agent distil
python -m selfplay_lab.train --resume --checkpoint checkpoints/latest.pt --episodes 12000
```

Each eval prints one JSON row. A 6,000-episode run of any variant on 3-player Leduc
takes about 15 seconds on a laptop CPU.

## The fly connectome arm, and its result

A reservoir of 449 recurrently connected nonlinear units will improve a weak
learner's features whether or not its wiring came from an animal. So "the
fly-brain agent beat the raw baseline" is not evidence about connectomes. The
question is whether the **measured** wiring beats a null that preserves
everything about it except the thing under test:

| null | holds fixed | tests |
|---|---|---|
| `rewired` | in/out degree per neuron, signs, weight multiset | which neuron connects to which — **the primary control** |
| `shuffled_signs` | topology, weights, density | where excitation and inhibition sit |
| `shuffled_weights` | topology, signs | whether graded synapse counts matter |
| `random_matched` | size, weight distribution | Erdos-Renyi topology; the weakest null |

```bash
python -m selfplay_lab.experiments.connectome_null --seeds 8 --episodes 6000
```

### Result: no detectable effect

3-player Leduc, DistilNFSP, 6,000 episodes, 8 seeds, identical hyperparameters
and identical evaluation deals across arms. Mean return against uniform random,
fair line 0:

| arm | mean return |
|---|---|
| `shuffled_weights` | **2.837** ± 0.247 |
| `random_matched` | 2.799 ± 0.418 |
| `rewired` | 2.753 ± 0.448 |
| `real` | 2.663 ± 0.366 |
| `raw` | 2.608 ± 0.272 |
| `shuffled_signs` | 2.551 ± 0.605 |

The measured connectome places **fourth of six**, below three of its own null
models. Paired by seed:

| paired comparison | mean | 95% CI (t) | 95% CI (bootstrap) | seeds favouring real | sign test |
|---|---|---|---|---|---|
| `real - rewired` vs random | −0.090 | [−0.583, +0.404] | [−0.416, +0.340] | 2 / 8 | p = 0.29 |
| head to head, same deals | −0.357 | [−0.919, +0.205] | [−0.808, +0.094] | 4 / 8 | p = 1.00 |

Every interval spans zero, under both a parametric and a distribution-free
estimate. **On this task the measured wiring is indistinguishable from a random
rewiring of itself.**

Head to head the split is 4 of 8 — not merely "not significant" but exactly what
the null predicts, at the largest p-value the sign test can return.

The distribution-free columns and a correction to the power calculation came out
of reading microsoft/SkillOpt; see [`docs/SKILLOPT_ASSESSMENT.md`](docs/SKILLOPT_ASSESSMENT.md)
for what was taken, what was refused, and the arithmetic that was wrong.

Note that `raw` is not a clean comparison: it is 47-dimensional where the fly arms
are 449-dimensional into the same network, so it differs in parameter count as
well as in features. `real` versus `rewired` is the clean one, and it is null.

### Why the null is a weaker statement than it looks

Reporting the null alone would overclaim it. `reservoir_diagnostics.py` measures
how much difference was available to find:

```
 steps  gain   linear R2      corr(real,rewired)
     1   4.0      0.8555                  1.0000
    12   4.0      0.8441                  0.9415   <- used by connectome_null.py
    24   4.0      0.8338                  0.8998
    40   8.0      0.7809                  0.7308
```

At the operating point actually used, **84% of the feature variance is an affine
function of the input**, and features from the real and rewired wirings correlate
at **0.94**. The reservoir is behaving mostly as a fixed random projection with a
mild nonlinearity; there was very little wiring information in the features for
any learner to exploit, whatever its capacity.

So both halves are the result: the experiment found no effect, *and* it had
limited power to find one, for a reason that is measured rather than guessed. The
design would have caught an effect of about **0.78** mean return 80% of the time;
smaller ones are not ruled out. (That figure was first published as 0.89, from a
power calculation that used the wrong t quantile at the wrong degrees of freedom
— the correction is in [`docs/SKILLOPT_ASSESSMENT.md`](docs/SKILLOPT_ASSESSMENT.md).
It erred toward claiming less sensitivity than the design had.)

### Two things this does not say

It does not say connectome-derived structure is useless in general. Two specific
weaknesses are named and testable:

1. **The reservoir is reset on every observation**, so its recurrence never
   accumulates anything - it settles from zero each time. Leduc's information
   state already carries the full history, so there was nothing for a dynamical
   system to add. A game where hidden information builds up across a hand is the
   regime where a stateful reservoir has something a feedforward net does not.
2. **The circuit does not match the task.** A ring attractor represents a
   circular variable; poker has none. The mushroom body - the fly's own
   associative-learning circuit, whose job is "does this cue predict good or
   bad" - is the better-matched substrate, and it is not what was tested here.

If those come back null too, the honest conclusion is that this does not help.

## What this release does NOT claim

- **No claim that the fly connectome helps.** It was tested and it did not, on
  3-player Leduc, against its own degree-preserving rewiring, on two metrics, at
  8 seeds. See the section above, including the measured reasons that null is a
  weaker statement than it first appears.
- **No strength numbers.** 6,000 episodes with default hyperparameters sits at the
  fair line on Leduc, which is what "learned nothing yet" looks like. The point of
  this release is that the machinery runs and the gates pass, not that it is tuned.
  Tuning and the real measurements were done on the private game and are not
  transferable claims.
- **No engine, no probes, no oracle.** Those depend on the private game.
- The leaf projection in `SearchNFSP` assumes a winner-take-all payoff (+v to the
  mover, -v shared by the other N-1 seats). For graded payoffs, override
  `NetGuidedEvaluator.evaluate`.

## Design rules carried over

- **Every gate asserts a proposition, not an event count.** A searcher that quietly
  ran UCT instead of PUCT, or stored the wrong distribution, produces a training run
  that looks completely normal. So the tests check the bot's configuration and the
  shape of the outputs directly.
- **Report the confound with the result.** Both variants change the self-play
  distribution as well as the SL target. That is the treatment, not a defect, but it
  means neither arm is a clean isolation of the objective, and the module docstrings
  say so.
- **A number that cannot be re-derived decays into a claim.** Eval seeds are fixed
  and printed; net forwards are counted so the cost model is auditable.

## Requirements

Python 3.10+, `open_spiel`, `torch`, `numpy`, `dm-tree`. See `requirements.txt`.
OpenSpiel's PyTorch NFSP/DQN internals are reached into deliberately (`_avg_network`,
`_rl_agent._q_network`, `_reservoir_buffer`); the constructors assert those attributes
exist so a moved internal fails at construction, not forty hours into a run.

## License

MIT. See `LICENSE`.
