# Search-in-training (factor T, E41): NFSP whose best-response step is a
# NET-GUIDED, LEAF-EVALUATED ISMCTS, and whose SL target is the search's visit
# distribution.
#
# WHAT "NET-GUIDED, LEAF-EVALUATED" MEANS, because the two words are two
# separate knobs and the bot we already have turns neither:
#
#   guided    the PRIOR that decides which branches get the thinking time.
#             A plain ISMCTS baseline uses a uniform prior and UCT, which
#             ignores the prior entirely. Here the prior is the seat's own
#             average-policy net and selection is PUCT, which multiplies by it
#             — so a better policy makes a better search.
#   leaf-eval how a leaf is scored. A rollout evaluator plays the imagined
#             game to the end. Here the leaf is scored by the seat's Q head and
#             the search stops there. This is the half that makes the arm
#             affordable at all: in the originating lab the rollout was the
#             measured cost centre.
#
# Together that is AlphaZero's shape, mapped onto NFSP's two heads:
#   AZ policy head  <->  the AVERAGE network  (trained here on visit counts)
#   AZ value head   <->  the inner DQN's Q    (trained, as always, on returns)
# The loop closes on the policy side: prior -> search -> better policy -> the
# SL target -> prior. The value side keeps learning from real self-play returns
# through machinery this module does not touch.
#
# THE ONE LEVER, stated so the confound is reported WITH the result (the E27
# convention): in BEST_RESPONSE mode the agent now (a) PLAYS the action the
# search chose rather than the epsilon-greedy one, and (b) stores the search's
# visit distribution as the SL label instead of the epsilon-greedy
# distribution. (a) changes the self-play distribution as well as the target,
# so like E27 this arm cannot separate "the SL objective" from "the states
# self-play visits". That is the treatment, not a defect — but it is not a
# clean isolation of the objective and must never be reported as one.
#
# Origin: extracted from a private self-play lab where this was experiment E41
# (factor T); E-numbers in this file refer to that lab's research log. Its gate
# is tests/test_search_nfsp.py.

import numpy as np
import torch

from open_spiel.python import rl_agent
from open_spiel.python.algorithms.ismcts import (
    ChildSelectionPolicy,
    ISMCTSBot,
    ISMCTSFinalPolicyType,
)
from open_spiel.python.pytorch import nfsp

MODE = nfsp.MODE

# A Q head trained on terminal-only +-1 with discount 1.0 estimates a return in
# [-1, 1]; anything outside is the net being wrong, not the game paying more.
VALUE_CLIP = 1.0


def softmax_over_legal(logits, legal, temperature):
    """[A] logits + list of legal actions -> [(action, prob)] over the legal
    set only. Uniform when the spread is degenerate, which is also the
    cold-start case (an untrained net separates nothing)."""
    z = np.asarray([logits[a] for a in legal], dtype=np.float64)
    if temperature <= 0:
        raise ValueError('temperature must be > 0')
    z = z - z.max()
    e = np.exp(np.clip(z / temperature, -60.0, 60.0))
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        return [(a, 1.0 / len(legal)) for a in legal]
    return [(a, float(p)) for a, p in zip(legal, e / s)]


class NetGuidedEvaluator:
    """ISMCTS evaluator: prior from the average net, leaf value from the Q head.

    ONE forward per leaf, for the seat to move — which is the cost model every
    projection is priced on (e.g. 64 sims x 250 decisions = 16,000
    forwards/game). The N-forward variant (evaluate every seat with its own
    net and centre) is a registered alternative, not this."""

    def __init__(self, agents, prior_temp=1.0):
        self._agents = agents
        self._prior_temp = float(prior_temp)
        self.forwards = 0          # counted so the cost model is auditable

    def _infostate(self, state, pid):
        return torch.tensor(
            np.asarray(state.information_state_tensor(pid), dtype=np.float32),
            dtype=torch.float32).unsqueeze(0)

    def prior(self, state):
        pid = state.current_player()
        legal = state.legal_actions(pid)
        if len(legal) <= 1:
            return [(a, 1.0) for a in legal]
        agent = self._agents[pid]
        with torch.no_grad():
            logits = agent._avg_network(self._infostate(state, pid))[0]
        self.forwards += 1
        return softmax_over_legal(logits.numpy(), legal, self._prior_temp)

    def evaluate(self, state):
        """Leaf value -> an N-vector of returns (N = num_players).

        The Q head scores the seat TO MOVE. Projecting one seat's value onto
        every seat is a modelling choice and it is registered as one: the seat
        gets +v and the other N-1 seats share -v equally. That is the shape of
        a winner-take-all reward (+1 to the winner, -1 to everyone else); for
        games with a different payoff shape, override evaluate()."""
        pid = state.current_player()
        legal = state.legal_actions(pid)
        agent = self._agents[pid]
        with torch.no_grad():
            q = agent._rl_agent._q_network(self._infostate(state, pid))[0]
        self.forwards += 1
        v = float(np.max([q[a].item() for a in legal])) if legal else 0.0
        v = max(-VALUE_CLIP, min(VALUE_CLIP, v))
        n = state.num_players()
        out = np.full(n, -v / max(1, n - 1), dtype=np.float64)
        out[pid] = v
        return out


class SearchNFSP(nfsp.NFSP):
    """NFSP whose BEST_RESPONSE step searches. See the module docstring."""

    def __init__(self, *args, search_sims=64, search_every=1,
                 prior_temp=1.0, uct_c=2.0, dirichlet_frac=0.25,
                 dirichlet_alpha=1.0, **kwargs):
        self._search_sims = int(search_sims)
        self._search_every = max(1, int(search_every))
        self._prior_temp = float(prior_temp)
        self._uct_c = float(uct_c)
        self._dirichlet_frac = float(dirichlet_frac)
        self._dirichlet_alpha = float(dirichlet_alpha)
        self._env = None
        self._all_agents = None
        self._bot = None
        self._evaluator = None
        self._search_calls = 0
        self._search_skipped_forced = 0
        self._search_skipped_sampled = 0
        super().__init__(*args, **kwargs)
        # Fail LOUDLY at construction on an open_spiel whose internals moved,
        # rather than 40 hours into a run. Same guard DistilNFSP carries, plus
        # the two this class reaches for that it does not.
        for attr in ('_mode', '_prev_timestep', '_prev_action', '_iteration',
                     '_learn_every', '_reservoir_buffer', '_rl_agent',
                     '_avg_network', '_num_actions'):
            assert hasattr(self, attr), (
                'installed open_spiel NFSP has no %s — its internals moved; '
                're-derive SearchNFSP.step against the installed source '
                'before training' % attr)
        assert hasattr(self._rl_agent, '_q_network'), (
            'installed open_spiel DQN has no _q_network — see above')
        assert hasattr(self._rl_agent, '_prev_action'), (
            'installed open_spiel DQN has no _prev_action — SearchNFSP must '
            'correct it after overriding the played action, or the Q learner '
            'trains on a transition that never happened')

    # ---- wiring done once, by the trainer, before the first episode -------
    def attach(self, env, agents, seed):
        """Give the agent the live state and the sibling nets the evaluator
        needs. Explicit rather than discovered: an agent that silently found
        no state would fall back to stock NFSP and train a DIFFERENT arm under
        this arm's folder name."""
        self._env = env
        self._all_agents = agents
        self._evaluator = NetGuidedEvaluator(agents, self._prior_temp)
        self._bot = ISMCTSBot(
            game=env.game,
            evaluator=self._evaluator,
            uct_c=self._uct_c,
            max_simulations=self._search_sims,
            random_state=np.random.RandomState(seed),
            # NORMALIZED_VISITED_COUNT is the AlphaZero target: the policy IS
            # the visit distribution. MAX_VISIT_COUNT (the OpenSpiel default)
            # collapses it to a one-hot and would throw away everything the
            # search learned about second choices.
            final_policy_type=ISMCTSFinalPolicyType.NORMALIZED_VISITED_COUNT,
            # PUCT, not UCT: UCT ignores `prior` entirely, so with UCT this
            # searcher would be leaf-evaluated but NOT guided — half the arm,
            # silently. tests/test_search_nfsp.py asserts this is PUCT.
            child_selection_policy=ChildSelectionPolicy.PUCT,
        )

    def searching(self):
        return self._search_sims > 0 and self._bot is not None

    # ---- exploration (standing design rule 4) ----------------------------
    def _noisy_root_policy(self, policy, rng):
        """Dirichlet noise on the ROOT visit distribution.

        Alpha is a parameter with a derivation and not AlphaZero's 0.03: that
        value is tuned for Go's ~362 legal moves via the alpha ~ 10/n rule of
        thumb. A card game offers a handful of legal actions at a typical
        decision, so 10/n lands around 1, and copying 0.03 would make the noise
        a near-deterministic one-hot — the opposite of exploration."""
        if self._dirichlet_frac <= 0 or len(policy) < 2:
            return policy
        acts = [a for a, _ in policy]
        ps = np.array([p for _, p in policy], dtype=np.float64)
        noise = rng.dirichlet([self._dirichlet_alpha] * len(acts))
        mixed = (1.0 - self._dirichlet_frac) * ps + self._dirichlet_frac * noise
        s = mixed.sum()
        if s <= 0:
            return policy
        return list(zip(acts, mixed / s))

    def step(self, time_step, is_evaluation=False):
        """Stock NFSP.step with ONE branch changed, kept line-for-line parallel
        to the installed source so a future open_spiel diff is easy to re-apply.

        Everything except the BEST_RESPONSE act is delegated to super()."""
        if (is_evaluation or time_step.last() or not self.searching()
                or self._mode != MODE.BEST_RESPONSE):
            return super().step(time_step, is_evaluation)

        legal = time_step.observations['legal_actions'][self.player_id]
        if len(legal) < 2:
            # A forced move needs no search, and searching one would spend a
            # net forward to rediscover the only legal action.
            self._search_skipped_forced += 1
            return super().step(time_step, is_evaluation)
        if self._search_every > 1 and (self._iteration % self._search_every):
            self._search_skipped_sampled += 1
            return super().step(time_step, is_evaluation)

        state = self._env.get_state
        # The DQN still takes its own step, so its replay buffer, epsilon
        # schedule and learning are exactly as before...
        rl_out = self._rl_agent.step(time_step, is_evaluation)
        policy, action = self._bot.step_with_policy(state)
        policy = self._noisy_root_policy(policy, self._bot._random_state)
        acts = [a for a, _ in policy]
        ps = np.array([p for _, p in policy], dtype=np.float64)
        ps = ps / ps.sum()
        action = int(self._bot._random_state.choice(acts, p=ps))
        # ...but it must be told which action was ACTUALLY played, or its next
        # transition records (s, a_epsilon, s') for an s' that came from the
        # searched action. Reaching into the inner agent is deliberate and
        # asserted at construction.
        self._rl_agent._prev_action = action
        self._search_calls += 1

        probs = np.zeros(self._num_actions, dtype=np.float32)
        for a, p in zip(acts, ps):
            probs[a] = p
        agent_output = rl_agent.StepOutput(action=action, probs=probs)
        self.add_transition(time_step, agent_output)
        del rl_out

        # --- the tail of NFSP.step, unchanged ---
        self._iteration += 1
        if self._iteration % self._learn_every == 0 and self._reservoir_buffer:
            self._last_sl_loss_value = self._learn()
        self._prev_timestep = time_step
        self._prev_action = agent_output.action
        return agent_output

    def search_stats(self):
        return {
            'search_calls': self._search_calls,
            'skipped_forced': self._search_skipped_forced,
            'skipped_sampled': self._search_skipped_sampled,
            'net_forwards': (self._evaluator.forwards
                             if self._evaluator else 0),
        }
