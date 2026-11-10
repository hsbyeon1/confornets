"""Align-free atom-pair distance diversity loss and guidance atom-set parsing."""

import json

import biotite.structure as struc
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from confornet.core.atom_sets import build_pair_index, load_guidances, parse_atom_set
from confornet.core.losses import pairwise_distance_mse_loss, pairwise_mse_loss


def _rigid(x, seed=1):
    R = torch.tensor(Rotation.random(random_state=seed).as_matrix(), dtype=x.dtype)
    return x @ R.T + torch.tensor([5.0, -3.0, 7.0], dtype=x.dtype)


@pytest.fixture
def coords():
    torch.manual_seed(0)
    return torch.randn(120, 3) * 10


@pytest.fixture
def pairs():
    idx = torch.arange(40, 60)
    ii, jj = torch.triu_indices(len(idx), len(idx), 1)
    return torch.stack([idx[ii], idx[jj]])


def test_rigid_invariance(coords, pairs):
    xl = torch.stack([coords, _rigid(coords), _rigid(coords, 7)])
    assert abs(float(pairwise_distance_mse_loss(xl, pairs))) < 1e-6
    assert pairwise_distance_mse_loss(xl[:, None], pairs) == pairwise_distance_mse_loss(xl, pairs)


def test_known_value():
    a = torch.tensor([[0.0, 0, 0], [3.0, 0, 0]])
    b = torch.tensor([[0.0, 0, 0], [5.0, 0, 0]])           # distance 3 -> 5
    pi = torch.tensor([[0], [1]])
    assert float(pairwise_distance_mse_loss(torch.stack([a, b]), pi)) == pytest.approx(-4.0)
    c = torch.tensor([[0.0, 0, 0], [3.0, 0, 0]])
    # three structures: pairs (a,b)=4, (a,c)=0, (b,c)=4 -> mean 8/3
    assert float(pairwise_distance_mse_loss(torch.stack([a, b, c]), pi)) == pytest.approx(-8 / 3)


def test_outside_atoms_ignored(coords, pairs):
    other = coords.clone(); other[:30] += 3.0
    assert abs(float(pairwise_distance_mse_loss(torch.stack([coords, other]), pairs))) < 1e-8


def test_gradient_only_on_pair_atoms(coords, pairs):
    moved = coords.clone(); moved[45:50, 0] += 2.0
    xl = torch.stack([coords, moved]).requires_grad_(True)
    pairwise_distance_mse_loss(xl, pairs).backward()
    hit = torch.nonzero(xl.grad.abs().sum(-1))
    assert set(hit[:, 1].tolist()) <= set(range(40, 60))
    assert set(hit[:, 0].tolist()) == {0, 1}                 # both structures get gradient


def test_bf16_and_single(coords, pairs):
    moved = coords.clone(); moved[45:50, 0] += 2.0
    out = pairwise_distance_mse_loss(torch.stack([coords, moved]).to(torch.bfloat16), pairs)
    assert out.dtype == torch.bfloat16 and float(out) < 0
    assert float(pairwise_distance_mse_loss(coords[None], pairs)) == 0.0


def test_default_loss_untouched(coords):
    xl = torch.stack([coords, coords + 0.5 * torch.randn_like(coords)])
    assert float(pairwise_mse_loss(xl)) < 0


def _toy_array(n_res=10):
    atoms = []
    for r in range(1, n_res + 1):
        for nm in ("N", "CA", "C", "O"):
            atoms.append(struc.Atom([0.0, 0, 0], chain_id="A", res_id=r, res_name="ALA", atom_name=nm, element=nm[0]))
    atoms.append(struc.Atom([0.0, 0, 0], chain_id="B", res_id=1, res_name="LIG", atom_name="C1", element="C", hetero=True))
    return struc.array(atoms)


def test_parse_half_open():
    specs = parse_atom_set([["A", "3:6", "CA"], ["A", 8, "N"]])
    assert [(s.res_id, s.atom_name) for s in specs] == [(3, "CA"), (4, "CA"), (5, "CA"), (8, "N")]
    with pytest.raises(ValueError):
        parse_atom_set([["A", "3-6", "CA"]])
    with pytest.raises(ValueError):
        parse_atom_set([["A", True, "CA"]])


def test_build_pair_index_cliques_and_union():
    aa = _toy_array()
    g = [{"particle": {"atom_sets": [[["A", "1:4", "CA"]], [["A", "3:5", "CA"], ["B", 1, "C1"]]]}}]
    pi = build_pair_index(aa, g)
    ca = {r: int(np.where((aa.res_id == r) & (aa.atom_name == "CA") & (aa.chain_id == "A"))[0][0]) for r in range(1, 11)}
    lig = int(np.where(aa.chain_id == "B")[0][0])
    want = {tuple(sorted(p)) for p in [(ca[1], ca[2]), (ca[1], ca[3]), (ca[2], ca[3]),
                                       (ca[3], ca[4]), (ca[3], lig), (ca[4], lig)]}
    assert {tuple(p) for p in pi.T.tolist()} == want         # (ca3, ca4) etc. counted once
    assert build_pair_index(aa, None) is None
    with pytest.raises(ValueError):
        build_pair_index(aa, [{"particle": {"atom_sets": [[["A", "99:100", "CA"], ["A", 1, "CA"]]]}}])
    with pytest.raises(ValueError):
        build_pair_index(aa, [{"other": {}}])
    with pytest.raises(ValueError):
        build_pair_index(aa, [{"particle": {"atom_sets": [[["A", 1, "CA"]]]}}])


def test_load_guidances_forms(tmp_path):
    g = [{"particle": {"atom_sets": [[["A", "1:3", "CA"]]]}}]
    for i, payload in enumerate((g, {"guidances": g}, [{"name": "x", "guidances": g}])):
        f = tmp_path / f"g{i}.json"; f.write_text(json.dumps(payload))
        assert load_guidances(f) == g
    bad = tmp_path / "bad.json"; bad.write_text(json.dumps({"x": 1}))
    with pytest.raises(ValueError):
        load_guidances(bad)


def test_atom_set_pairs_cross_only():
    aa = _toy_array()
    ca = {r: int(np.where((aa.res_id == r) & (aa.atom_name == "CA") & (aa.chain_id == "A"))[0][0]) for r in range(1, 11)}
    g = [{"particle": {"atom_set_pairs": [[[["A", "8:10", "CA"]], [["A", "1:4", "CA"]]]]}}]
    got = {tuple(p) for p in build_pair_index(aa, g).T.tolist()}
    want = {tuple(sorted((ca[a], ca[b]))) for a in (8, 9) for b in (1, 2, 3)}
    assert got == want                                       # 6 cross pairs, no 1-2 / 8-9 pairs
    both = [{"particle": {"atom_sets": [[["A", "1:3", "CA"]]], "atom_set_pairs": g[0]["particle"]["atom_set_pairs"]}}]
    assert {tuple(p) for p in build_pair_index(aa, both).T.tolist()} == want | {(ca[1], ca[2])}
    with pytest.raises(ValueError):
        build_pair_index(aa, [{"particle": {"atom_set_pairs": [[[["A", 1, "CA"]]]]}}])
