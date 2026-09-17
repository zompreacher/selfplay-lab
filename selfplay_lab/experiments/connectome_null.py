"""Does the measured connectome do anything a rewiring of it does not?

This is the experiment the whole fly-brain arm stands or falls on. A reservoir of
449 recurrently connected nonlinear units improves a weak learner's features
whether or not its wiring came from an animal, so "the connectome agent beat the
raw baseline" is not evidence about connectomes. The question is whether the
MEASURED wiring beats a null that preserves everything about it except the thing
under test.

Arms, all trained with identical hyperparameters, identical agent seeds and
identical evaluation deals:

  raw               no reservoir; the OpenSpiel information state as shipped.
  real              the measured hemibrain central complex.
  rewired           degree-preserving rewiring. THE PRIMARY CONTROL.
  shuffled_signs    which neurons are inhibitory is permuted.
  shuffled_weights  synapse counts permuted across the same edges.
  random_matched    Erdos-Renyi topology, matched size and weight distribution.

Two measurements, because one of them is weak:

  vs-random   mean return against uniform-random opponents. Cheap, and what the
              repository already reports, but beating a random opponent mostly
              measures "does not fold at random" and saturates early.
  head-to-head  arm A's agents seated against arm B's agents on the same deals.
              Sharper, and the one to believe when the two disagree.

Seeds are paired: arm `real` at seed 3 and arm `rewired` at seed 3 share agent
initialisation and evaluation deals, so their difference is a paired sample and
the sign consistency across seeds is meaningful on its own.

Usage:
  python -m selfplay_lab.experiments.connectome_null --seeds 5 --episodes 6000
"""

import argparse
import json
import math
import os
import statistics
import time

import numpy as np
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nfsp

from .. import train as T

ARMS = ("raw", "real", "rewired", "shuffled_signs", "shuffled_weights", "random_matched")


def _args_for(arm, seed, base):
    args = argparse.Namespace(**vars(base))
    args.seed = seed
    args.features = "raw" if arm == "raw" else "fly"
    args.null = "real" if arm == "raw" else arm
    args.fly_seed = seed
    return args


def train_arm(game, arm, seed, base):
    """Train one arm at one seed. Returns (agents, feature map, final eval)."""
    args = _args_for(arm, seed, base)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    env = rl_environment.Environment(game, include_full_state=True)
    features, meta = T.build_features(env, args)
    agents = T.build_agents(env, args, features)
    for ep in range(1, args.episodes + 1):
        env.seed(seed * 1000003 + ep)
        T.play_episode(env, agents, features)
    ev = T.eval_vs_random(game, agents, args.eval_games, args.eval_seed, features)
    return agents, features, ev, meta


def head_to_head(game, side_a, side_b, games_per_seat, seed_base):
    """Mean return for side A, rotating A through every seat against B.

    Both sides use their own feature map on the same underlying game state, which
    is the point: the only asymmetry between them is the reservoir each one looks
    through. Deals are seeded so any two calls score the same panel.
    """
    a_agents, a_features = side_a
    b_agents, b_features = side_b
    n = game.num_players()
    env = rl_environment.Environment(game, include_full_state=False)
    returns = []
    for seat in range(n):
        total = 0.0
        for g in range(games_per_seat):
            env.seed(seed_base + seat * 100003 + g)
            ts = env.reset()
            contexts = [
                a_agents[seat].temp_mode_as(nfsp.MODE.AVERAGE_POLICY),
                *[b_agents[p].temp_mode_as(nfsp.MODE.AVERAGE_POLICY) for p in range(n)],
            ]
            for ctx in contexts:
                ctx.__enter__()
            try:
                while not ts.last():
                    pid = ts.observations["current_player"]
                    if pid == seat:
                        act = a_agents[seat].step(
                            a_features.transform(ts), is_evaluation=True
                        ).action
                    else:
                        act = b_agents[pid].step(
                            b_features.transform(ts), is_evaluation=True
                        ).action
                    ts = env.step([act])
            finally:
                for ctx in reversed(contexts):
                    ctx.__exit__(None, None, None)
            total += ts.rewards[seat]
        returns.append(total / games_per_seat)
    return float(np.mean(returns))


def summarise(values):
    """Mean, spread, and every underlying value.

    The per-seed list is not optional detail. A mean and a standard deviation
    that cannot be re-derived from the numbers behind them decays into a claim,
    and with five seeds the individual values carry most of the information
    about whether a difference is real.
    """
    values = [float(v) for v in values]
    summary = {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else None,
        "n": len(values),
        "per_seed": [round(v, 4) for v in values],
    }
    if len(values) > 1:
        summary["sem"] = round(statistics.stdev(values) / math.sqrt(len(values)), 4)
    return summary


def paired_difference(a_values, b_values):
    """Paired difference with a 95% interval, for arms that share seeds.

    Reported rather than a bare p-value: with n=5 the interval says what the
    experiment could and could not have detected, which is the part that matters
    when the answer comes back null.
    """
    diffs = [a - b for a, b in zip(a_values, b_values)]
    n = len(diffs)
    mean = statistics.mean(diffs)
    if n < 2:
        return {"mean": round(mean, 4), "n": n}
    sd = statistics.stdev(diffs)
    sem = sd / math.sqrt(n)
    # Student t, two-tailed, 95%, for small n. Table rather than a dependency on
    # scipy for four numbers.
    t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
              8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    # Smallest true effect this design would detect 80% of the time.
    t_power = {2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476, 6: 1.440, 7: 1.415,
               8: 1.397, 9: 1.383, 10: 1.372}.get(n, 1.282)
    return {
        "mean": round(mean, 4),
        "std": round(sd, 4),
        "sem": round(sem, 4),
        "ci95": [round(mean - t_crit * sem, 4), round(mean + t_crit * sem, 4)],
        "t": round(mean / sem, 3) if sem else None,
        "n": n,
        "favouring_first": sum(1 for d in diffs if d > 0),
        "min_detectable_effect_80pct": round((t_crit + t_power) * sem, 4),
        "per_seed": [round(d, 4) for d in diffs],
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--game", default="leduc_poker(players=3)")
    p.add_argument("--agent", choices=["stock", "distil"], default="distil")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=6000)
    p.add_argument("--h2h-games", type=int, default=400, help="per seat")
    p.add_argument("--h2h-seed", type=int, default=900000)
    p.add_argument("--out", default=None, help="write the full result as JSON")
    # training hyperparameters, identical for every arm
    p.add_argument("--hidden", default="128,128")
    p.add_argument("--optimizer", default="adam", choices=["sgd", "adam"])
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--reservoir", type=int, default=200000)
    p.add_argument("--replay", type=int, default=50000)
    p.add_argument("--anticipatory", type=float, default=0.1)
    p.add_argument("--min-buffer", type=int, default=1000)
    p.add_argument("--learn-every", type=int, default=64)
    p.add_argument("--distil-temp", type=float, default=0.5)
    p.add_argument("--eval-games", type=int, default=300)
    p.add_argument("--eval-seed", type=int, default=500000)
    p.add_argument("--extract", default=T.DEFAULT_EXTRACT)
    p.add_argument("--fly-steps", type=int, default=12)
    p.add_argument("--fly-train-dynamics", action="store_true")
    args = p.parse_args(argv)

    game = pyspiel.load_game(args.game)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = list(range(args.seeds))

    trained = {}
    vs_random = {arm: [] for arm in arms}
    t0 = time.time()
    for arm in arms:
        for seed in seeds:
            agents, features, ev, meta = train_arm(game, arm, seed, args)
            trained[(arm, seed)] = (agents, features)
            vs_random[arm].append(ev["pooled_return"])
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "seed": seed,
                        "vs_random_return": round(ev["pooled_return"], 4),
                        "vs_random_winrate": round(ev["pooled"], 4),
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                ),
                flush=True,
            )

    result = {
        "config": {
            "game": args.game,
            "agent": args.agent,
            "episodes": args.episodes,
            "seeds": args.seeds,
            "optimizer": args.optimizer,
            "lr": args.lr,
        },
        "vs_random_return": {arm: summarise(vs_random[arm]) for arm in arms},
    }

    # Head-to-head against the primary control, paired by seed.
    if "real" in arms and "rewired" in arms:
        paired = []
        for seed in seeds:
            score = head_to_head(
                game,
                trained[("real", seed)],
                trained[("rewired", seed)],
                args.h2h_games,
                args.h2h_seed,
            )
            paired.append(score)
            print(
                json.dumps({"head_to_head": "real_vs_rewired", "seed": seed,
                            "real_mean_return": round(score, 4)}),
                flush=True,
            )
        result["head_to_head_real_vs_rewired"] = {
            **paired_difference(paired, [0.0] * len(paired)),
            "per_seed": [round(x, 4) for x in paired],
            "note": "positive means the measured wiring beat its degree-preserving rewiring",
        }
        if "real" in arms and "rewired" in arms:
            result["vs_random_real_minus_rewired"] = paired_difference(
                vs_random["real"], vs_random["rewired"]
            )

    print("\n=== vs uniform random (mean return, fair line 0) ===")
    for arm in arms:
        s = result["vs_random_return"][arm]
        std = f"+/- {s['std']}" if s["std"] is not None else ""
        print(f"  {arm:<18} {s['mean']:>8.4f} {std}")
    if "head_to_head_real_vs_rewired" in result:
        h = result["head_to_head_real_vs_rewired"]
        print("\n=== head to head: real vs rewired (fair line 0) ===")
        print(f"  real mean return {h['mean']:+.4f} +/- {h['std']}")
        print(f"  95% CI [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]"
              f"   {'SPANS ZERO - no detectable effect' if h['ci95'][0] < 0 < h['ci95'][1] else ''}")
        print(f"  seeds favouring real: {h['favouring_first']}/{len(seeds)}")
        print(f"  smallest effect this design would reliably detect: "
              f"{h['min_detectable_effect_80pct']:.2f}")
        print(f"  per seed: {h['per_seed']}")

    if args.out:
        # Create the directory rather than discovering it is missing after the
        # runs have finished: this experiment costs ~25 minutes of compute and
        # losing it to a mkdir is an unforced error.
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {args.out}")
    return result


if __name__ == "__main__":
    main()
