# BR-distillation NFSP (E27): the SL/average net learns the CURRENT best
# response instead of a reservoir of its historical samples.
#
# Stock NFSP stores (state, action-probs) pairs at the moment the inner RL
# learner acted, then trains the average/deployable net on uniform samples of
# that history — an average whose labels are frozen at insertion time. E18
# measured the consequence (the reservoir fills at ~ep32k and new play enters
# with probability falling like 1/n), E20 falsified the buffer-arithmetic fix,
# E21 confirmed the ~30pp average-vs-BR gap on nine arms, and E22 showed the
# gap is recoverable from the Q head at EXPORT time. This class asks the
# remaining constructive question: what happens when the deployable net is
# trained on the live Q head's policy all along?
#
# THE ONE LEVER: `_learn()`'s LABELS. A batch is still drawn from the same
# reservoir (same states, same coverage), the loss / optimizer / schedule are
# untouched — but the target distribution is recomputed from the CURRENT
# Q network at every SL step instead of read from storage. Side effects that
# are part of the treatment and must be reported with it: (1) the average-
# policy MODE of self-play now plays ~the current soft BR, so the opponent
# mixture during training changes too — this arm cannot separate the SL
# objective from the self-play distribution (E22 already isolated the pure
# export question); (2) SL labels no longer contain epsilon-exploration noise
# (stock stores the epsilon-greedy distribution; this computes the softened
# greedy policy).
#
# THE TARGET IS E22's OBJECT, EXACTLY: softmax over the state's LEGAL Q values
# standardised to mean 0 / spread 1 (population std), at temperature
# --distil-temp (default 0.5 = the graded br-T0.5). The standardisation is the
# whole design — a raw softmax(Q/T) repeats the E17 kill-rule mistake, because
# Adam's Q values live on a different scale than SGD's and T would mean a
# different policy on every run. Degenerate spread (the head is not separating
# the legal actions) falls back to uniform-over-legal.
#
# Origin: extracted from a private self-play lab where this was experiment E27;
# the E-numbers in this docstring refer to that lab's research log. The
# propositions the lab's parity test asserted (rows sum to 1, legal-only
# support, affine invariance, degenerate fallback) are tests/test_distil_targets.py.

import torch
import torch.nn.functional as F

from open_spiel.python.pytorch import dqn
from open_spiel.python.pytorch import nfsp

# Numerical guards for the standardised softmax.
MIN_STD = 1e-9
MAX_EXPONENT = 60.0


def distil_targets(q, legal, temperature):
    """Batched E22 br-T policy: [B,A] Q values + [B,A] bool legal mask ->
    [B,A] target probs (softmax over standardised legal Q at `temperature`;
    uniform over legal where the legal spread is degenerate or <2 actions).

    Population std over the LEGAL entries only."""
    if temperature <= 0:
        raise ValueError('temperature must be > 0')
    legal = legal.bool()
    n = legal.sum(dim=1, keepdim=True).clamp(min=1).to(q.dtype)
    q_legal = q.masked_fill(~legal, 0.0)
    mean = q_legal.sum(dim=1, keepdim=True) / n
    var = ((q - mean) ** 2).masked_fill(~legal, 0.0).sum(dim=1, keepdim=True) / n
    std = var.clamp(min=0.0).sqrt()
    z = (q - mean) / std.clamp(min=MIN_STD)
    scaled = (z / temperature).clamp(-MAX_EXPONENT, MAX_EXPONENT)
    scaled = scaled.masked_fill(~legal, -float('inf'))
    probs = F.softmax(scaled, dim=-1)
    uniform = legal.to(q.dtype) / n
    degenerate = (std < MIN_STD) | (legal.sum(dim=1, keepdim=True) < 2)
    return torch.where(degenerate, uniform, probs)


class DistilNFSP(nfsp.NFSP):
    """NFSP whose SL step distils the live Q head (see module docstring)."""

    def __init__(self, *args, distil_temp=0.5, **kwargs):
        if distil_temp <= 0:
            raise ValueError('--distil-temp must be > 0, got %r' % distil_temp)
        self._distil_temp = float(distil_temp)
        super().__init__(*args, **kwargs)
        # Fail LOUDLY on an open_spiel whose NFSP internals moved: every
        # attribute _learn() below reaches
        # for, checked at construction rather than 40 hours into a run.
        for attr in ('_reservoir_buffer', '_batch_size',
                     '_min_buffer_size_to_learn', '_sl_loss_fn', '_device',
                     '_avg_network', '_optimizer', '_gradient_norm_clipping'):
            assert hasattr(self, attr), (
                'installed open_spiel NFSP has no %s — its internals moved; '
                're-derive DistilNFSP._learn against the installed source '
                'before training' % attr)
        assert hasattr(self._rl_agent, '_q_network'), (
            'installed open_spiel DQN has no _q_network — see above')

    def _learn(self):
        """Stock NFSP._learn with ONE change: the targets are recomputed from
        the current Q network instead of read from the stored transitions.
        Structure kept line-for-line parallel to the stock method so a future
        open_spiel diff is easy to re-apply."""
        if (len(self._reservoir_buffer) < self._batch_size
                or len(self._reservoir_buffer) < self._min_buffer_size_to_learn):
            return None

        transitions = self._reservoir_buffer.sample(self._batch_size)

        info_states = torch.tensor(
            transitions.info_state, device=self._device, dtype=torch.float32)
        legal_actions_mask = torch.tensor(
            transitions.legal_actions_mask, device=self._device,
            dtype=torch.bool)

        # THE LEVER: live labels. (Stock reads transitions.action_probs here.)
        with torch.no_grad():
            q_values = self._rl_agent._q_network(info_states)
            action_probs = distil_targets(
                q_values, legal_actions_mask, self._distil_temp).detach()

        avg_actions_logits = self._avg_network(info_states)
        avg_actions_logits = torch.where(
            legal_actions_mask,
            avg_actions_logits,
            torch.full_like(avg_actions_logits,
                            dqn.ILLEGAL_ACTION_LOGITS_PENALTY),
        )

        loss = self._sl_loss_fn(avg_actions_logits, action_probs).mean()
        self._optimizer.zero_grad()
        loss.backward()

        if self._gradient_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                self._avg_network.parameters(), self._gradient_norm_clipping)
        self._optimizer.step()

        return loss.item()
