"""The connectome as a fixed recurrent reservoir.

    r_{t+1} = r_t + (dt/tau) * (clip(gain * W r_t + background + I, 0, 1) - r_t)

`W` is the measured connectome and is a buffer, not a parameter: it never
receives a gradient. Training all 63,417 measured weights would destroy the
measurement and leave an ordinarily-initialised RNN with an unusual topology,
which is a different (and much less interesting) object.

Per-neuron `gain` and `tau` ARE parameters, defaulting to frozen. An adjacency
table genuinely does not measure a neuron's intrinsic excitability or membrane
time constant, so training those does not overwrite anything that was measured.
Turn them on with `trainable_dynamics=True`.

Implemented in torch rather than numpy specifically so that option exists.
"""

import numpy as np
import torch
from torch import nn

# Integration and operating-point constants, carried over from the ai-gm
# connectome module where they were chosen by a scan over gain and background
# drive. They are NOT derived from the connectome: the wiring fixes what
# connects to what, not where the neurons sit on their input-output curve.
DT_OVER_TAU = 0.1
DEFAULT_GAIN = 4.0
BACKGROUND_DRIVE = 0.2
MAX_RATE = 1.0

# Steps the reservoir is run per observation. Long enough for the recurrence to
# matter (several multiples of dt/tau), short enough that a self-play run is not
# dominated by it.
DEFAULT_STEPS = 12

# Fraction of neurons each input feature projects onto. Sparse rather than dense
# so that different features drive distinguishable subpopulations instead of all
# of them, which is what gives the recurrence something to separate.
INPUT_DENSITY = 0.10


class FlyReservoir(nn.Module):
    """Map an observation vector to the connectome's settled activity."""

    def __init__(
        self,
        weights,
        input_dim,
        *,
        seed=0,
        steps=DEFAULT_STEPS,
        gain=DEFAULT_GAIN,
        background=BACKGROUND_DRIVE,
        dt_over_tau=DT_OVER_TAU,
        max_rate=MAX_RATE,
        input_scale=1.0,
        input_density=INPUT_DENSITY,
        trainable_dynamics=False,
    ):
        super().__init__()
        matrix = torch.as_tensor(np.asarray(weights), dtype=torch.float32)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"weights must be square, got {tuple(matrix.shape)}")
        self.n_neurons = matrix.shape[0]
        self.input_dim = int(input_dim)
        self.steps = int(steps)
        self.background = float(background)
        self.dt_over_tau = float(dt_over_tau)
        self.max_rate = float(max_rate)

        # The connectome. A buffer, so it moves with the module and is saved in
        # the state dict, but never appears in .parameters().
        self.register_buffer("W", matrix)

        # Fixed sparse random input projection. Seeded from its own generator so
        # that every arm of an experiment - real and null alike - gets the exact
        # same encoder, and the recurrent matrix is the only thing that differs.
        generator = torch.Generator().manual_seed(seed)
        dense = torch.randn(self.n_neurons, self.input_dim, generator=generator)
        mask = (
            torch.rand(self.n_neurons, self.input_dim, generator=generator) < input_density
        ).float()
        self.register_buffer("encoder", dense * mask * float(input_scale))

        log_gain = torch.full((self.n_neurons,), float(np.log(gain)))
        log_dt = torch.full((self.n_neurons,), float(np.log(dt_over_tau)))
        if trainable_dynamics:
            self.log_gain = nn.Parameter(log_gain)
            self.log_dt = nn.Parameter(log_dt)
        else:
            self.register_buffer("log_gain", log_gain)
            self.register_buffer("log_dt", log_dt)
        self.trainable_dynamics = bool(trainable_dynamics)

    @property
    def output_dim(self):
        return self.n_neurons

    def forward(self, x):
        """(batch, input_dim) -> (batch, n_neurons) settled rates.

        The reservoir is reset for every observation. That makes it a pure
        function of its input - which is what lets the feature map be cached, and
        what keeps the real-versus-null comparison free of order effects. For a
        game where hidden information accumulates across a hand, carry state
        instead; Leduc's information state already contains the full history.
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)
        drive = x @ self.encoder.t()
        gain = torch.exp(self.log_gain)
        step = torch.exp(self.log_dt)

        rates = torch.zeros(x.shape[0], self.n_neurons, dtype=torch.float32, device=x.device)
        for _ in range(self.steps):
            recurrent = rates @ self.W.t()
            target = torch.clamp(gain * recurrent + self.background + drive, 0.0, self.max_rate)
            rates = rates + step * (target - rates)
        return rates.squeeze(0) if single else rates
