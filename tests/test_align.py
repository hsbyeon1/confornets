"""Tests for kabsch_align against OF3's weighted_rigid_align."""

import torch
from openfold3.core.loss.diffusion import weighted_rigid_align

from confornet.utils.align import kabsch_align


def test_kabsch_align_matches_of3():
    """Test that kabsch_align produces same results as OF3's weighted_rigid_align."""
    torch.manual_seed(42)

    n_atoms = 50
    mobile = torch.randn(n_atoms, 3)
    target = torch.randn(n_atoms, 3)

    # Our implementation
    aligned_ours = kabsch_align(mobile, target)

    # OF3 implementation (needs batch dim, weights, mask)
    mobile_batched = mobile.unsqueeze(0)
    target_batched = target.unsqueeze(0)
    weights = torch.ones(1, n_atoms)
    mask = torch.ones(1, n_atoms)

    aligned_of3 = weighted_rigid_align(
        mobile_batched, target_batched, weights, mask, eps=1e-8
    ).squeeze(0)

    diff = (aligned_ours - aligned_of3).abs().max().item()
    assert diff < 1e-5, f"Results differ by {diff}"


def test_kabsch_align_gradient_flow():
    """Test that gradients flow through kabsch_align (unlike OF3 which detaches)."""
    torch.manual_seed(42)

    n_atoms = 50
    mobile = torch.randn(n_atoms, 3, requires_grad=True)
    target = torch.randn(n_atoms, 3)

    aligned = kabsch_align(mobile, target)
    loss = aligned.sum()
    loss.backward()

    assert mobile.grad is not None
    assert not torch.isnan(mobile.grad).any()
    assert (mobile.grad.abs() > 0).any()
