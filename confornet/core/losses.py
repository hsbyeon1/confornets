import torch
import torch.nn.functional as F

from confornet.utils.align import kabsch_align


def aligned_mse_loss(x: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    """MSE loss after rigid alignment (aligns x_target to x)."""
    x_target_aligned = kabsch_align(x_target, x).detach()
    return F.mse_loss(x, x_target_aligned)


def pairwise_mse_loss(xl_all: torch.Tensor) -> torch.Tensor:
    """
    Mean pairwise MSE between k structures for diversity maximization.
    Returns NEGATIVE for maximization.
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
            mse_ij = aligned_mse_loss(xl_flat[i], xl_flat[j])
            mse_ji = aligned_mse_loss(xl_flat[j], xl_flat[i])
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
