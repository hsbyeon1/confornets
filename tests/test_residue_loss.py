"""Residue-specific (masked) coord_mse diversity loss."""

import biotite.structure as struc
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from confornet.core.losses import (
    aligned_mse_loss,
    pairwise_mse_loss,
    parse_resid_ranges,
    select_loss_atoms,
)
from confornet.utils.align import kabsch_align


def _kabsch_align_original(mobile, target):
    """kabsch_align as it was before kabsch_transform was factored out."""
    mobile_c = mobile - mobile.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    H = target_c.T @ mobile_c
    U, _, Vh = torch.linalg.svd(H.float())
    d = torch.sign(torch.linalg.det(U @ Vh))
    F = torch.eye(3, device=mobile.device, dtype=U.dtype)
    F[2, 2] = d
    R = U @ F @ Vh
    return (mobile_c.float() @ R.T + target.mean(dim=0, keepdim=True)).to(mobile.dtype)


@pytest.fixture
def coords():
    torch.manual_seed(0)
    return torch.randn(200, 3) * 10


def _rigid(x, seed=1):
    R = torch.tensor(Rotation.random(random_state=seed).as_matrix(), dtype=x.dtype)
    return x @ R.T + torch.tensor([5.0, -3.0, 7.0], dtype=x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kabsch_align_unchanged(coords, dtype):
    a, b = coords.to(dtype), (_rigid(coords) + 0.3 * torch.randn_like(coords)).to(dtype)
    assert torch.equal(kabsch_align(a, b), _kabsch_align_original(a, b))


def test_default_loss_unchanged(coords):
    xl = torch.stack([coords, coords + 0.5 * torch.randn_like(coords), _rigid(coords, 3)])
    flat = xl
    ref = []
    for i in range(3):
        for j in range(i + 1, 3):
            for a, b in ((flat[i], flat[j]), (flat[j], flat[i])):
                ref.append(torch.nn.functional.mse_loss(a, _kabsch_align_original(b, a).detach()))
    assert torch.allclose(pairwise_mse_loss(xl), -torch.stack(ref).mean())
    assert torch.equal(pairwise_mse_loss(xl[:, None]), pairwise_mse_loss(xl))


def test_full_index_sets_match_default(coords):
    xl = torch.stack([coords, coords + 0.5 * torch.randn_like(coords)])
    allidx = torch.arange(coords.shape[0])
    assert torch.allclose(pairwise_mse_loss(xl, allidx, allidx), pairwise_mse_loss(xl), atol=1e-5)


def test_masked_rigid_invariance(coords):
    frame, loss = torch.arange(0, 60), torch.arange(150, 160)
    xl = torch.stack([coords, _rigid(coords)])
    assert abs(float(pairwise_mse_loss(xl, frame, loss))) < 1e-8


def test_masked_shift_and_leakage(coords):
    frame, loss = torch.arange(0, 60), torch.arange(150, 160)
    moved = coords.clone(); moved[loss, 0] += 2.0          # loss atoms move 2 A in x
    assert float(pairwise_mse_loss(torch.stack([coords, moved]), frame, loss)) == pytest.approx(-4 / 3, abs=1e-4)
    other = coords.clone(); other[100:140] += 2.0          # neither frame nor loss atoms
    assert abs(float(pairwise_mse_loss(torch.stack([coords, other]), frame, loss))) < 1e-8


def test_masked_gradient_only_on_loss_atoms(coords):
    frame, loss = torch.arange(0, 60), torch.arange(150, 160)
    moved = coords.clone(); moved[loss, 0] += 2.0
    xl = torch.stack([coords, moved]).requires_grad_(True)
    pairwise_mse_loss(xl, frame, loss).backward()
    nz = set(torch.nonzero(xl.grad.abs().sum(-1))[:, 1].tolist())
    assert nz == set(loss.tolist())


def test_masked_bf16(coords):
    frame, loss = torch.arange(0, 60), torch.arange(150, 160)
    moved = coords.clone(); moved[loss, 0] += 2.0
    out = pairwise_mse_loss(torch.stack([coords, moved]).to(torch.bfloat16), frame, loss)
    assert out.dtype == torch.bfloat16
    assert float(out) == pytest.approx(-4 / 3, abs=2e-2)


def test_aligned_mse_loss_frame_only(coords):
    frame = torch.arange(0, 60)
    assert abs(float(aligned_mse_loss(coords, _rigid(coords), frame_idx=frame))) < 1e-8


def test_parse_resid_ranges():
    assert parse_resid_ranges("3-5,10,12-13") == [3, 4, 5, 10, 12, 13]
    assert parse_resid_ranges(" 7 , 7-8 ") == [7, 8]


def _toy_array():
    names = ["N", "CA", "C", "O", "H"]
    n_res = 6
    atoms = []
    for r in range(1, n_res + 1):
        for nm in names:
            atoms.append(struc.Atom([0.0, 0.0, 0.0], chain_id="A", res_id=r, res_name="ALA",
                                    atom_name=nm, element=nm[0]))
    atoms.append(struc.Atom([0.0, 0.0, 0.0], chain_id="B", res_id=1, res_name="LIG",
                            atom_name="CA", element="C", hetero=True))
    return struc.array(atoms)


def test_select_loss_atoms():
    aa = _toy_array()
    ca = select_loss_atoms(aa, [2, 3], "ca", "A")
    assert aa.atom_name[ca].tolist() == ["CA", "CA"] and aa.res_id[ca].tolist() == [2, 3]
    heavy = select_loss_atoms(aa, [2], "heavy", "A")
    assert sorted(aa.atom_name[heavy].tolist()) == ["C", "CA", "N", "O"]
    assert len(select_loss_atoms(aa, [1], "ca", None)) == 2       # chain B ligand atom included
    with pytest.raises(ValueError):
        select_loss_atoms(aa, [99], "ca", "A")
