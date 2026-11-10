import numpy as np
import torch
import torch.nn.functional as F

from confornet.utils.align import kabsch_align, kabsch_transform


def aligned_mse_loss(
    x: torch.Tensor,
    x_target: torch.Tensor,
    frame_idx: torch.Tensor | None = None,
    loss_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE loss after rigid alignment (aligns x_target to x).

    frame_idx: atom indices used to compute the superposition (default: all atoms).
    loss_idx:  atom indices the MSE is taken over (default: all atoms).
    With neither given this is the original whole-structure loss. The aligned
    target is detached in every case, so gradient flows through x only.
    """
    if frame_idx is None and loss_idx is None:
        x_target_aligned = kabsch_align(x_target, x).detach()
        return F.mse_loss(x, x_target_aligned)

    fi = slice(None) if frame_idx is None else frame_idx
    li = slice(None) if loss_idx is None else loss_idx
    with torch.no_grad():
        R, mobile_mean, target_mean = kabsch_transform(x_target[fi].float(), x[fi].float())
        x_target_aligned = (x_target[li].float() - mobile_mean) @ R.T + target_mean
    return F.mse_loss(x[li].float(), x_target_aligned).to(x.dtype)


def parse_resid_ranges(spec: str) -> list[int]:
    """'245-252', '3-24,43-64,122-139' or '101' -> sorted residue numbers (inclusive)."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, hi = part.split("-") if "-" in part else (part, part)
        out.update(range(int(lo), int(hi) + 1))
    return sorted(out)


def select_loss_atoms(
    atom_array,
    resids: list[int],
    atoms: str = "ca",
    chain: str | None = "A",
) -> np.ndarray:
    """Atom indices (into the query atom_array / coordinate tensor) for residues `resids`.

    atoms: 'ca' (C-alpha only) or 'heavy' (all non-hydrogen atoms).
    Residues are matched on atom_array.res_id (the numbering written to the output
    CIF) and, unless chain is None, atom_array.chain_id. Raises if any residue is absent.
    """
    if atoms not in ("ca", "heavy"):
        raise ValueError(f"atoms must be 'ca' or 'heavy', got {atoms!r}")
    sel = np.isin(atom_array.res_id, resids)
    if chain is not None:
        sel &= atom_array.chain_id == chain
    sel &= (atom_array.atom_name == "CA") if atoms == "ca" else (atom_array.element != "H")
    idx = np.where(sel)[0]
    missing = sorted(set(resids) - set(atom_array.res_id[idx].tolist()))
    if missing:
        raise ValueError(f"residues not found on chain {chain}: {missing}")
    return idx


def pairwise_mse_loss(
    xl_all: torch.Tensor,
    frame_idx: torch.Tensor | None = None,
    loss_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mean pairwise MSE between k structures for diversity maximization.
    Returns NEGATIVE for maximization.

    frame_idx / loss_idx: optional atom subsets for the per-pair superposition and
    for the MSE (see aligned_mse_loss). Default: all atoms for both.
    """
    k = xl_all.shape[0]
    if k < 2:
        return torch.tensor(0.0, device=xl_all.device, dtype=xl_all.dtype)

    # Flatten [k, S, N_atom, 3] -> [k, N_atom, 3] for S=1
    if xl_all.dim() == 4 and xl_all.shape[1] == 1:
        xl_flat = xl_all.squeeze(1)
    else:
        xl_flat = xl_all.reshape(k, -1, 3)

    total_mse = torch.tensor(0.0, device=xl_all.device, dtype=xl_all.dtype)
    n_pairs = 0

    for i in range(k):
        for j in range(i + 1, k):
            mse_ij = aligned_mse_loss(xl_flat[i], xl_flat[j], frame_idx, loss_idx)
            mse_ji = aligned_mse_loss(xl_flat[j], xl_flat[i], frame_idx, loss_idx)
            total_mse = total_mse + mse_ij + mse_ji
            n_pairs += 2

    return -total_mse / n_pairs  # Negative for maximization


def pairwise_tensor_mse(tensors: torch.Tensor) -> torch.Tensor:
    """Pairwise MSE between k tensors [k, ...]. Returns NEGATIVE for maximization."""
    k = tensors.shape[0]
    if k < 2:
        return torch.tensor(0.0, device=tensors.device, dtype=tensors.dtype)

    flat = tensors.reshape(k, -1)
    total_mse = torch.tensor(0.0, device=tensors.device, dtype=tensors.dtype)
    n_pairs = 0

    for i in range(k):
        for j in range(i + 1, k):
            total_mse = total_mse + F.mse_loss(flat[i], flat[j])
            n_pairs += 1

    return -total_mse / n_pairs


def distogram_to_cdf(logits: torch.Tensor) -> torch.Tensor:
    """Softmax + cumsum along the bin axis. Returns CDF in [0, 1]."""
    return F.softmax(logits, dim=-1).cumsum(dim=-1)
