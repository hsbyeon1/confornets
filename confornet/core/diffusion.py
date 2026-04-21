import torch
import torch.nn as nn

# Re-exported so callers can `from confornet.core.diffusion import create_noise_schedule`.
from openfold3.core.model.structure.diffusion_module import create_noise_schedule  # noqa: F401


def _center(xl: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask[..., None]
    mean = torch.sum(xl * m, dim=-2, keepdim=True) / torch.sum(
        m, dim=-2, keepdim=True
    ).clamp(min=1e-8)
    return (xl - mean) * m


def diffusion_sample(
    diffusion_module: nn.Module,
    batch: dict,
    si_input: torch.Tensor,
    si_trunk: torch.Tensor,
    zij_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    no_samples: int,
    step_scale: float = 1.0,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deterministic DDIM rollout for differentiable ConforNet-training steps."""
    atom_mask = batch["atom_mask"]
    batch_dim = atom_mask.shape[0]
    num_atoms = atom_mask.shape[-1]
    device = atom_mask.device
    dtype = si_input.dtype

    if initial_noise is not None:
        xl = noise_schedule[0] * initial_noise
    else:
        xl = noise_schedule[0] * torch.randn(
            (batch_dim, no_samples, num_atoms, 3), device=device, dtype=dtype
        )

    mask = atom_mask
    if mask.dim() == 3 and mask.shape[1] == 1:
        mask = mask.expand(-1, no_samples, -1)

    for tau in range(len(noise_schedule) - 1):
        t_current = noise_schedule[tau]
        t_next = noise_schedule[tau + 1]

        xl = _center(xl, mask)

        t_tensor = torch.full(
            (batch_dim, no_samples), t_current.item(), device=device, dtype=dtype
        )

        xl_denoised = diffusion_module(
            batch=batch,
            xl_noisy=xl,
            token_mask=batch["token_mask"],
            atom_mask=atom_mask,
            t=t_tensor,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_conditioning=True,
            use_high_precision_attention=True,
            _mask_trans=True,
        )

        delta = (xl - xl_denoised) / t_current
        xl = xl + step_scale * (t_next - t_current) * delta

    return xl
