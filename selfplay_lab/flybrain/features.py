"""Presenting reservoir activity to an agent in place of the raw observation.

The integration point is deliberately the *observation*, not the agent. NFSP
agents here only ever read `timestep.observations["info_state"][player]`, so
swapping that vector for the connectome's response leaves every agent, optimiser
and buffer in the repository untouched. Real and null arms then differ in exactly
one object: the recurrent matrix inside the reservoir.

Caching is exact, not approximate. `FlyReservoir.forward` resets its state every
call, so it is a pure function of the observation, and two identical information
states must produce identical features. Leduc has few enough distinct
information states that the cache turns the reservoir from the dominant cost
into a rounding error.
"""

import numpy as np
import torch


class ReservoirFeatures:
    """Replace `info_state` with the reservoir's settled activity."""

    def __init__(self, reservoir, *, cache=True):
        self.reservoir = reservoir
        self.output_dim = reservoir.output_dim
        self._cache = {} if cache else None
        self.hits = 0
        self.misses = 0

    def encode(self, info_state):
        vector = np.asarray(info_state, dtype=np.float32)
        if self._cache is None:
            with torch.no_grad():
                return self.reservoir(vector).numpy()
        key = vector.tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        with torch.no_grad():
            features = self.reservoir(vector).numpy()
        self._cache[key] = features
        return features

    def transform(self, timestep):
        """Return a TimeStep whose `info_state` entries are reservoir features.

        Every player's observation is mapped, not just the acting player's:
        agents read their own `info_state` on terminal steps too, when it is
        their turn to do bookkeeping rather than to act.
        """
        observations = dict(timestep.observations)
        observations["info_state"] = [
            self.encode(state) for state in timestep.observations["info_state"]
        ]
        return timestep._replace(observations=observations)

    def stats(self):
        total = self.hits + self.misses
        return {
            "distinct_info_states": len(self._cache) if self._cache is not None else None,
            "cache_hit_rate": round(self.hits / total, 4) if total else None,
        }


class IdentityFeatures:
    """The `--features raw` arm: no reservoir at all.

    Present so that the baseline runs through the same wrapper as every other
    arm. A control that takes a different code path is not a control.
    """

    def __init__(self, output_dim):
        self.output_dim = output_dim

    def encode(self, info_state):
        return np.asarray(info_state, dtype=np.float32)

    def transform(self, timestep):
        return timestep

    def stats(self):
        return {"distinct_info_states": None, "cache_hit_rate": None}
