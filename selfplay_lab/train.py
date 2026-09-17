"""Generic NFSP self-play trainer for any OpenSpiel game.

Three agent variants, selected by --agent:

  stock    open_spiel's NFSP as shipped.
  distil   DistilNFSP (nfsp_distil.py): the average net is trained on the
           live Q head's softened best response instead of the reservoir's
           frozen labels.
  search   SearchNFSP (nfsp_search.py): the best-response step is a
           net-guided, leaf-evaluated ISMCTS (PUCT), and the SL target is
           the search's visit distribution. Needs a game that implements
           resample_from_infostate (leduc_poker, kuhn_poker, and most
           imperfect-information games in OpenSpiel do).

One agent per seat. Evaluation rotates the trained agent through every
seat against uniform-random opponents on a FIXED seed base, so two
checkpoints are always scored on the same deals (common random numbers).

Usage:
  python -m selfplay_lab.train --game "leduc_poker(players=3)" --episodes 2000
  python -m selfplay_lab.train --agent distil --episodes 2000
  python -m selfplay_lab.train --agent search --search-sims 16 --episodes 500
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nfsp

from .flybrain import FlyReservoir, apply_null, load_extract
from .flybrain.features import IdentityFeatures, ReservoirFeatures
from .nfsp_distil import DistilNFSP
from .nfsp_search import SearchNFSP

DEFAULT_EXTRACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fixtures", "connectome", "ring_attractor.hemibrain-v1.2.json.gz",
)


def build_features(env, args):
    """Choose what the agents actually see.

    `raw` is the untouched OpenSpiel information state. `fly` replaces it with a
    measured fly connectome's response to that state, with `--null` selecting
    whether the connectome is the real one or a control. Both arms go through the
    same wrapper so that nothing but the recurrent matrix differs between them.
    """
    info_size = env.observation_spec()["info_state"][0]
    if args.features == "raw":
        return IdentityFeatures(info_size), {"features": "raw", "input_dim": info_size}
    extract = load_extract(args.extract)
    weights, null_info = apply_null(extract, args.null, seed=args.fly_seed)
    reservoir = FlyReservoir(
        weights,
        input_dim=info_size,
        seed=args.fly_seed,
        steps=args.fly_steps,
        trainable_dynamics=args.fly_train_dynamics,
    )
    meta = {
        "features": "fly",
        "input_dim": info_size,
        "reservoir_dim": reservoir.output_dim,
        "circuit": extract.circuit,
        "n_neurons": extract.n_neurons,
        "n_edges": extract.n_edges,
        "citation": extract.provenance.get("citation"),
        **null_info,
    }
    return ReservoirFeatures(reservoir), meta


def build_agents(env, args, features):
    info_size = features.output_dim
    num_actions = env.action_spec()["num_actions"]
    n = env.num_players
    common = dict(
        state_representation_size=info_size,
        num_actions=num_actions,
        hidden_layers_sizes=[int(h) for h in args.hidden.split(",")],
        reservoir_buffer_capacity=args.reservoir,
        anticipatory_param=args.anticipatory,
        batch_size=args.batch_size,
        rl_learning_rate=args.lr,
        sl_learning_rate=args.lr,
        min_buffer_size_to_learn=args.min_buffer,
        learn_every=args.learn_every,
        optimizer_str=args.optimizer,
        # DQN kwargs pass through NFSP's **kwargs
        replay_buffer_capacity=args.replay,
        epsilon_decay_duration=args.episodes * 10,
    )
    agents = []
    for pid in range(n):
        seed = args.seed + pid
        if args.agent == "stock":
            a = nfsp.NFSP(pid, seed=seed, **common)
        elif args.agent == "distil":
            a = DistilNFSP(pid, seed=seed, distil_temp=args.distil_temp, **common)
        elif args.agent == "search":
            a = SearchNFSP(pid, seed=seed, search_sims=args.search_sims,
                           search_every=args.search_every,
                           prior_temp=args.prior_temp, uct_c=args.uct_c,
                           dirichlet_frac=args.dirichlet_frac,
                           dirichlet_alpha=args.dirichlet_alpha, **common)
        else:
            raise ValueError(args.agent)
        agents.append(a)
    if args.agent == "search":
        for a in agents:
            a.attach(env, agents, seed=args.seed + 1000 + a.player_id)
    return agents


def play_episode(env, agents, features=None):
    """`features=None` means the agents see the raw information state.

    Defaulted rather than required so that callers which predate the reservoir -
    including the existing gates - keep working unchanged.
    """
    features = features or IdentityFeatures(env.observation_spec()["info_state"][0])
    ts = env.reset()
    while not ts.last():
        pid = ts.observations["current_player"]
        out = agents[pid].step(features.transform(ts))
        ts = env.step([out.action])
    for a in agents:
        a.step(features.transform(ts))  # terminal bookkeeping
    return ts.rewards


def eval_vs_random(game, agents, games_per_seat, seed_base, features=None):
    """Average-policy win rate AND mean return, one focal agent per seat,
    others uniform random.

    Deals are seeded from `seed_base` so any two evaluations play the same
    panel (common random numbers). Win rate is the winner-take-all view and
    its fair line is 1/N; mean return is the right metric for games with
    graded payoffs (poker), where the fair line is 0.
    """
    n = game.num_players()
    env = rl_environment.Environment(game, include_full_state=False)
    features = features or IdentityFeatures(env.observation_spec()["info_state"][0])
    rng = np.random.RandomState(seed_base)
    per_seat, per_seat_return = [], []
    for seat in range(n):
        wins, ret = 0, 0.0
        for g in range(games_per_seat):
            env.seed(seed_base + seat * 100003 + g)
            ts = env.reset()
            with agents[seat].temp_mode_as(nfsp.MODE.AVERAGE_POLICY):
                while not ts.last():
                    pid = ts.observations["current_player"]
                    if pid == seat:
                        act = agents[seat].step(
                            features.transform(ts), is_evaluation=True
                        ).action
                    else:
                        legal = ts.observations["legal_actions"][pid]
                        act = int(rng.choice(legal))
                    ts = env.step([act])
            r = ts.rewards
            ret += r[seat]
            if r[seat] == max(r) and list(r).count(max(r)) == 1:
                wins += 1
        per_seat.append(wins / games_per_seat)
        per_seat_return.append(ret / games_per_seat)
    return {"per_seat": per_seat, "pooled": float(np.mean(per_seat)),
            "per_seat_return": per_seat_return,
            "pooled_return": float(np.mean(per_seat_return)),
            "fair_line": 1.0 / n, "games_per_seat": games_per_seat}


def save_checkpoint(path, agents, episode, history, args):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "episode": episode,
        "history": history,
        "args": vars(args),
        "agents": [
            {"avg": a._avg_network.state_dict(),
             "q": a._rl_agent._q_network.state_dict()}
            for a in agents
        ],
    }
    torch.save(payload, path)


def load_checkpoint(path, agents):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for a, st in zip(agents, payload["agents"]):
        a._avg_network.load_state_dict(st["avg"])
        a._rl_agent._q_network.load_state_dict(st["q"])
    return payload["episode"], payload["history"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="leduc_poker(players=3)")
    p.add_argument("--agent", choices=["stock", "distil", "search"], default="stock")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", default="128,128")
    p.add_argument("--reservoir", type=int, default=200000)
    p.add_argument("--replay", type=int, default=50000)
    p.add_argument("--anticipatory", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--min-buffer", type=int, default=1000)
    p.add_argument("--learn-every", type=int, default=64)
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    # fly connectome reservoir
    p.add_argument("--features", choices=["raw", "fly"], default="raw",
                   help="raw OpenSpiel information state, or a fly connectome's response to it")
    p.add_argument("--null", default="real",
                   choices=["real", "rewired", "shuffled_signs", "shuffled_weights",
                            "random_matched"],
                   help="which wiring to use; 'rewired' is the primary control")
    p.add_argument("--extract", default=DEFAULT_EXTRACT)
    p.add_argument("--fly-steps", type=int, default=12)
    p.add_argument("--fly-seed", type=int, default=0)
    p.add_argument("--fly-train-dynamics", action="store_true",
                   help="train per-neuron gain and tau; synapse weights stay frozen either way")
    # distil
    p.add_argument("--distil-temp", type=float, default=0.5)
    # search
    p.add_argument("--search-sims", type=int, default=16)
    p.add_argument("--search-every", type=int, default=1)
    p.add_argument("--prior-temp", type=float, default=1.0)
    p.add_argument("--uct-c", type=float, default=2.0)
    p.add_argument("--dirichlet-frac", type=float, default=0.25)
    p.add_argument("--dirichlet-alpha", type=float, default=1.0)
    # eval / io
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-games", type=int, default=200, help="per seat")
    p.add_argument("--eval-seed", type=int, default=500000)
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    game = pyspiel.load_game(args.game)
    # include_full_state=True is what lets SearchNFSP read env.get_state.
    env = rl_environment.Environment(game, include_full_state=True)

    if args.agent == "search" and args.features != "raw":
        # SearchNFSP runs ISMCTS over real game states and evaluates leaves with
        # its own nets. Feeding it reservoir features would train something other
        # than the experiment its name claims, silently. Refuse instead.
        raise SystemExit(
            "--agent search does not support --features fly: its ISMCTS evaluates "
            "leaf states directly, so the reservoir would be applied inconsistently."
        )

    features, feature_meta = build_features(env, args)
    agents = build_agents(env, args, features)
    if not args.quiet:
        print(json.dumps({"config": feature_meta}))

    start, history = 0, []
    if args.resume and os.path.exists(args.checkpoint):
        start, history = load_checkpoint(args.checkpoint, agents)
        if not args.quiet:
            print(f"resumed from {args.checkpoint} at episode {start}")

    t0 = time.time()
    for ep in range(start + 1, args.episodes + 1):
        env.seed(args.seed * 1000003 + ep)
        play_episode(env, agents, features)
        if ep % args.eval_every == 0 or ep == args.episodes:
            ev = eval_vs_random(game, agents, args.eval_games, args.eval_seed, features)
            row = {"episode": ep, "elapsed_s": round(time.time() - t0, 1),
                   "vs_random_pooled": round(ev["pooled"], 4),
                   "vs_random_per_seat": [round(x, 4) for x in ev["per_seat"]],
                   "vs_random_mean_return": round(ev["pooled_return"], 4),
                   "fair_line": round(ev["fair_line"], 4)}
            if args.agent == "search":
                row["search"] = agents[0].search_stats()
            if args.features != "raw":
                row["reservoir"] = features.stats()
            history.append(row)
            if not args.quiet:
                print(json.dumps(row))
            save_checkpoint(args.checkpoint, agents, ep, history, args)
    return history


if __name__ == "__main__":
    main()
