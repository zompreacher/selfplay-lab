"""Guards on DistilNFSP.distil_targets (the E22 br-T object, batched)."""
import torch

from selfplay_lab.nfsp_distil import distil_targets, MIN_STD


def test_rows_are_distributions_over_legal_only():
    q = torch.tensor([[1.0, 3.0, -2.0, 0.5], [0.0, 0.0, 5.0, 1.0]])
    legal = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=torch.bool)
    p = distil_targets(q, legal, temperature=0.5)
    assert torch.allclose(p.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.all(p[~legal] == 0)
    assert torch.all(p[legal] > 0)


def test_higher_q_gets_higher_prob_and_temperature_sharpens():
    q = torch.tensor([[1.0, 2.0, 3.0]])
    legal = torch.ones(1, 3, dtype=torch.bool)
    warm = distil_targets(q, legal, temperature=1.0)[0]
    cold = distil_targets(q, legal, temperature=0.1)[0]
    assert warm[0] < warm[1] < warm[2]
    assert cold[2] > warm[2]


def test_affine_invariance_is_the_design():
    # Standardisation means a*Q + b gives the SAME policy: the whole point
    # of the E22 object is that Adam's and SGD's Q scales agree on the policy.
    q = torch.tensor([[0.3, -1.2, 2.0, 0.7]])
    legal = torch.ones(1, 4, dtype=torch.bool)
    base = distil_targets(q, legal, 0.5)
    shifted = distil_targets(q * 37.0 + 11.0, legal, 0.5)
    assert torch.allclose(base, shifted, atol=1e-5)


def test_degenerate_spread_and_single_legal_fall_back_to_uniform():
    q = torch.tensor([[2.0, 2.0, 2.0], [9.0, -9.0, 4.0]])
    legal = torch.tensor([[1, 1, 1], [0, 1, 0]], dtype=torch.bool)
    p = distil_targets(q, legal, 0.5)
    assert torch.allclose(p[0], torch.full((3,), 1 / 3))
    assert torch.allclose(p[1], torch.tensor([0.0, 1.0, 0.0]))


def test_temperature_must_be_positive():
    q = torch.zeros(1, 2)
    legal = torch.ones(1, 2, dtype=torch.bool)
    try:
        distil_targets(q, legal, 0.0)
    except ValueError:
        return
    raise AssertionError("temperature <= 0 must raise")
