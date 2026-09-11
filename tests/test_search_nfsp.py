"""Gate for SearchNFSP: every failure mode of this arm is silent, so each
proposition is asserted directly rather than inferred from a training curve."""
import numpy as np
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.ismcts import ChildSelectionPolicy, ISMCTSFinalPolicyType

from selfplay_lab.nfsp_search import SearchNFSP, NetGuidedEvaluator, softmax_over_legal
from selfplay_lab import train as T


def _build(n_players=3):
    game = pyspiel.load_game(f"leduc_poker(players={n_players})")
    env = rl_environment.Environment(game, include_full_state=True)
    info = env.observation_spec()["info_state"][0]
    na = env.action_spec()["num_actions"]
    agents = [SearchNFSP(pid, search_sims=8, state_representation_size=info,
                         num_actions=na, hidden_layers_sizes=[32],
                         reservoir_buffer_capacity=1000, anticipatory_param=0.1,
                         min_buffer_size_to_learn=20, learn_every=8, seed=pid)
              for pid in range(n_players)]
    for a in agents:
        a.attach(env, agents, seed=100 + a.player_id)
    return game, env, agents


def test_bot_is_puct_with_visit_count_policy():
    _, _, agents = _build()
    bot = agents[0]._bot
    assert bot._child_selection_policy == ChildSelectionPolicy.PUCT
    assert bot._final_policy_type == ISMCTSFinalPolicyType.NORMALIZED_VISITED_COUNT


def test_search_actually_runs_and_forwards_are_counted():
    _, env, agents = _build()
    for ep in range(30):
        env.seed(ep)
        T.play_episode(env, agents)
    stats = [a.search_stats() for a in agents]
    assert sum(s["search_calls"] for s in stats) > 0
    assert sum(s["net_forwards"] for s in stats) > 0


def test_leaf_projection_matches_num_players():
    for n in (2, 3):
        game, env, agents = _build(n)
        ev = NetGuidedEvaluator(agents)
        st = game.new_initial_state()
        while st.is_chance_node():
            st.apply_action(st.chance_outcomes()[0][0])
        out = ev.evaluate(st)
        assert out.shape == (n,)
        assert abs(out.sum()) < 1e-9  # +v to mover, -v shared by the rest


def test_softmax_over_legal_is_a_distribution():
    logits = np.array([0.5, -3.0, 2.0, 1.0])
    pol = softmax_over_legal(logits, [0, 2, 3], 1.0)
    acts = [a for a, _ in pol]
    ps = [p for _, p in pol]
    assert acts == [0, 2, 3]
    assert abs(sum(ps) - 1.0) < 1e-9
    assert ps[1] > ps[2] > ps[0]
