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

## What this release does NOT claim

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
