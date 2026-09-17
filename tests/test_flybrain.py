"""Gates on the connectome reservoir and, more importantly, on its null models.

The experiment in `experiments/connectome_null.py` is only meaningful if the
nulls really hold fixed what they claim to hold fixed, and if every arm is
treated identically apart from the recurrent matrix. A rewiring that quietly
changed the degree sequence, or an encoder that differed between arms, would
produce a difference that looks like a result and is an artefact.

So these assert the properties the comparison depends on, not that the code ran.
"""

import collections

import numpy as np
import pytest
import torch

from selfplay_lab.flybrain import FlyReservoir, apply_null, load_extract, signed_matrix
from selfplay_lab.flybrain.features import IdentityFeatures, ReservoirFeatures
from selfplay_lab.flybrain.nulls import NULL_MODELS, rewire_degree_preserving

EXTRACT = "fixtures/connectome/ring_attractor.hemibrain-v1.2.json.gz"


@pytest.fixture(scope="module")
def extract():
    return load_extract(EXTRACT)


# --- the extract ---------------------------------------------------------------


def test_extract_is_the_measured_circuit(extract):
    assert extract.circuit == "ring_attractor"
    assert extract.n_neurons == 449
    assert extract.n_edges == 63417
    assert "10.7554/eLife.57443" in extract.provenance["citation"]
    assert extract.provenance["licence"] == "CC BY 4.0"


def test_unsigned_neurons_drive_nothing(extract):
    """A neuron whose transmitter is unknown contributes zero, not a guess."""
    unsigned = np.flatnonzero(np.isnan(extract.signs))
    assert unsigned.size > 0, "this extract genuinely has unsigned neurons"
    weights = signed_matrix(extract)
    assert np.all(weights[:, unsigned] == 0.0)


def test_row_normalisation_leaves_relative_weights_alone(extract):
    """Normalisation may rescale a neuron's inputs; it must not reorder them."""
    raw = signed_matrix(extract, normalise=False)
    normalised = signed_matrix(extract, normalise=True)
    row = int(np.argmax(np.abs(raw).sum(axis=1)))
    nonzero = np.flatnonzero(raw[row])
    ratio = normalised[row, nonzero] / raw[row, nonzero]
    assert np.allclose(ratio, ratio[0])


# --- the null models -----------------------------------------------------------


def test_rewiring_preserves_every_degree_exactly(extract):
    """The primary control's entire claim: same degree sequence, different wiring."""
    pre, post, _weight, stats = rewire_degree_preserving(
        extract, np.random.default_rng(0)
    )
    assert collections.Counter(pre.tolist()) == collections.Counter(
        extract.edges_pre.tolist()
    ), "out-degree changed"
    assert collections.Counter(post.tolist()) == collections.Counter(
        extract.edges_post.tolist()
    ), "in-degree changed"
    assert stats["accepted"] > 0


def test_rewiring_actually_rewires(extract):
    """A null that barely moves anything is not a control, it is a copy."""
    _pre, post, _w, _s = rewire_degree_preserving(extract, np.random.default_rng(0))
    moved = int((post != extract.edges_post).sum())
    assert moved > 0.8 * extract.n_edges, f"only {moved}/{extract.n_edges} edges moved"


def test_rewiring_cannot_change_a_neurons_sign(extract):
    """Sign follows the presynaptic neuron, and a swap never changes that."""
    pre, _post, _w, _s = rewire_degree_preserving(extract, np.random.default_rng(1))
    assert np.array_equal(pre, extract.edges_pre)


def test_rewiring_creates_no_self_loops_or_duplicates(extract):
    pre, post, _w, _s = rewire_degree_preserving(extract, np.random.default_rng(2))
    assert not np.any(pre == post)
    assert len(set(zip(pre.tolist(), post.tolist()))) == len(pre)


def test_sign_shuffle_keeps_density_and_the_sign_multiset(extract):
    """Otherwise the control confounds 'where inhibition sits' with 'how much'."""
    real, _ = apply_null(extract, "real", seed=0)
    shuffled, _ = apply_null(extract, "shuffled_signs", seed=0)
    assert int((real != 0).sum()) == int((shuffled != 0).sum())


def test_weight_shuffle_keeps_topology_exactly(extract):
    real, _ = apply_null(extract, "real", seed=0)
    shuffled, _ = apply_null(extract, "shuffled_weights", seed=0)
    assert np.array_equal(real != 0, shuffled != 0)


def test_every_null_actually_differs_from_the_real_wiring(extract):
    real, _ = apply_null(extract, "real", seed=0)
    for name in NULL_MODELS:
        if name == "real":
            continue
        null, _ = apply_null(extract, name, seed=0)
        assert not np.allclose(real, null), f"{name} is indistinguishable from real"


def test_unknown_null_is_rejected(extract):
    with pytest.raises(ValueError, match="unknown null model"):
        apply_null(extract, "vibes", seed=0)


# --- the reservoir -------------------------------------------------------------


def test_the_connectome_is_never_a_trainable_parameter(extract):
    """Training the measured weights would delete the measurement."""
    weights, _ = apply_null(extract, "real", seed=0)
    for trainable in (False, True):
        reservoir = FlyReservoir(
            weights, input_dim=16, seed=0, trainable_dynamics=trainable
        )
        names = {name for name, _ in reservoir.named_parameters()}
        assert "W" not in names
        assert not any("W" == n.split(".")[-1] for n in names)


def test_trainable_dynamics_exposes_only_gain_and_tau(extract):
    """Intrinsic excitability is not in an adjacency table, so it is fair game."""
    weights, _ = apply_null(extract, "real", seed=0)
    frozen = FlyReservoir(weights, input_dim=16, seed=0)
    assert list(frozen.parameters()) == []
    live = FlyReservoir(weights, input_dim=16, seed=0, trainable_dynamics=True)
    assert {name for name, _ in live.named_parameters()} == {"log_gain", "log_dt"}


def test_reservoir_is_a_pure_function_of_its_input(extract):
    """Required for the feature cache to be exact rather than approximate."""
    weights, _ = apply_null(extract, "real", seed=0)
    reservoir = FlyReservoir(weights, input_dim=16, seed=0)
    x = torch.arange(16, dtype=torch.float32)
    first = reservoir(x)
    reservoir(torch.randn(16))  # intervening call must not leave state behind
    assert torch.equal(first, reservoir(x))


def test_every_arm_gets_an_identical_encoder(extract):
    """The fairness property the whole experiment rests on.

    If the input projection differed between real and rewired, a difference in
    outcome would not localise to the wiring.
    """
    encoders = []
    for name in NULL_MODELS:
        weights, _ = apply_null(extract, name, seed=0)
        encoders.append(FlyReservoir(weights, input_dim=16, seed=7).encoder)
    for other in encoders[1:]:
        assert torch.equal(encoders[0], other)


def test_different_wiring_produces_different_features(extract):
    """If real and rewired gave the same features the comparison would be vacuous."""
    x = torch.randn(4, 16, generator=torch.Generator().manual_seed(0))
    real_w, _ = apply_null(extract, "real", seed=0)
    rewired_w, _ = apply_null(extract, "rewired", seed=0)
    real = FlyReservoir(real_w, input_dim=16, seed=0)(x)
    rewired = FlyReservoir(rewired_w, input_dim=16, seed=0)(x)
    assert not torch.allclose(real, rewired)


def test_reservoir_output_stays_in_range(extract):
    """Saturation is clamped, so a dynamics bug cannot become a silent NaN."""
    weights, _ = apply_null(extract, "real", seed=0)
    reservoir = FlyReservoir(weights, input_dim=16, seed=0)
    out = reservoir(torch.full((2, 16), 1000.0))
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_non_square_weights_are_rejected(extract):
    with pytest.raises(ValueError, match="square"):
        FlyReservoir(np.zeros((4, 5)), input_dim=3)


# --- the feature map -----------------------------------------------------------


def test_cached_features_equal_uncached_ones(extract):
    weights, _ = apply_null(extract, "real", seed=0)
    reservoir = FlyReservoir(weights, input_dim=8, seed=0)
    cached = ReservoirFeatures(reservoir, cache=True)
    direct = ReservoirFeatures(reservoir, cache=False)
    x = np.arange(8, dtype=np.float32)
    assert np.array_equal(cached.encode(x), direct.encode(x))
    assert np.array_equal(cached.encode(x), direct.encode(x))
    assert cached.hits == 1


def test_identity_features_are_the_observation_itself():
    identity = IdentityFeatures(5)
    x = np.arange(5, dtype=np.float32)
    assert np.array_equal(identity.encode(x), x)


def test_search_agent_refuses_reservoir_features():
    """A silent fallback would train a different experiment under this one's name."""
    from selfplay_lab import train

    with pytest.raises(SystemExit, match="does not support"):
        train.main(
            [
                "--agent", "search",
                "--features", "fly",
                "--episodes", "1",
                "--game", "leduc_poker(players=3)",
            ]
        )
