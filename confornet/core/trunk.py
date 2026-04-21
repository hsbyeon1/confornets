from enum import Enum

import torch
import torch.nn as nn
from openfold3.core.metrics.aggregate_confidence_ranking import get_confidence_scores


class ConforNetMode(str, Enum):
    PAIR_PRE = "pair_pre"
    PAIR_POST = "pair_post"


def run_trunk_with_confornet(
    model: nn.Module,
    batch: dict,
    confornet: nn.Module | None = None,
    mode: ConforNetMode = ConforNetMode.PAIR_PRE,
    num_recycles: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run trunk with optional ConforNet injection at last recycle.
    Returns: si_input, si_trunk, zij_trunk (all [B, N, ...])
    """
    num_cycles = num_recycles + 1
    mem = model._get_mode_mem_settings()

    with torch.no_grad():
        s_input, s_init, z_init = model.input_embedder(
            batch=batch, inplace_safe=False, use_high_precision_attention=True
        )
        token_mask = batch["token_mask"]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        for cycle_no in range(num_cycles - 1):
            s, z = _run_cycle(
                model, batch, s_input, s_init, z_init, s, z, token_mask, pair_mask, mem
            )

        s, z = _run_pre_pairformer(
            model, batch, s_input, s_init, z_init, s, z, pair_mask, mem
        )

    z = z.detach()
    s = s.detach()
    s_input = s_input.detach()

    if confornet is not None and mode == ConforNetMode.PAIR_PRE:
        z = confornet(z)

    s, z = model.pairformer_stack(
        s=s,
        z=z,
        single_mask=token_mask.to(dtype=z.dtype),
        pair_mask=pair_mask.to(dtype=s.dtype),
        chunk_size=mem.chunk_size,
        use_deepspeed_evo_attention=False,
        use_cueq_triangle_kernels=mem.use_cueq_triangle_kernels,
        use_lma=mem.use_lma,
        inplace_safe=False,
        _mask_trans=True,
    )

    if confornet is not None and mode == ConforNetMode.PAIR_POST:
        z = confornet(z)

    return s_input, s, z


def run_batched_trunk_k_confornets(
    model: nn.Module,
    batch: dict,
    confornets: list[nn.Module],
    num_recycles: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run trunk with k ConforNets batched (k as batch dim).
    ConforNet applied at PAIR_PRE position only.
    Returns: si_input, si_trunk, zij_trunk (all [k, 1, N, ...])
    """
    k = len(confornets)
    num_cycles = num_recycles + 1
    mem = model._get_mode_mem_settings()

    with torch.no_grad():
        s_input, s_init, z_init = model.input_embedder(
            batch=batch, inplace_safe=False, use_high_precision_attention=True
        )
        token_mask = batch["token_mask"]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        for cycle_no in range(num_cycles - 1):
            s, z = _run_cycle(
                model, batch, s_input, s_init, z_init, s, z, token_mask, pair_mask, mem
            )

        s, z = _run_pre_pairformer(
            model, batch, s_input, s_init, z_init, s, z, pair_mask, mem
        )

    z = z.detach()
    s = s.detach()
    s_input = s_input.detach()
    token_mask = token_mask.detach()
    pair_mask = pair_mask.detach()

    z_transformed_list = [confornet(z) for confornet in confornets]
    z_batched = torch.cat(z_transformed_list, dim=0)  # [k, N, N, c_z]
    s_batched = s.repeat(k, 1, 1)  # [k, N, c_s]
    token_mask_batched = token_mask.expand(k, -1)
    pair_mask_batched = pair_mask.expand(k, -1, -1)

    # Batched pairformer
    s_batched, z_batched = model.pairformer_stack(
        s=s_batched,
        z=z_batched,
        single_mask=token_mask_batched.to(dtype=z_batched.dtype),
        pair_mask=pair_mask_batched.to(dtype=s_batched.dtype),
        chunk_size=mem.chunk_size,
        use_deepspeed_evo_attention=False,
        use_cueq_triangle_kernels=mem.use_cueq_triangle_kernels,
        use_lma=mem.use_lma,
        inplace_safe=False,
        _mask_trans=True,
    )

    # Add sample dimension
    si_input = s_input.expand(k, -1, -1).unsqueeze(1)  # [k, 1, N, c_s_input]
    si_trunk = s_batched.unsqueeze(1)  # [k, 1, N, c_s]
    zij_trunk = z_batched.unsqueeze(1)  # [k, 1, N, N, c_z]

    return si_input, si_trunk, zij_trunk


def _run_pre_pairformer(
    model,
    batch,
    s_input,
    s_init,
    z_init,
    s,
    z,
    pair_mask,
    mem,
    use_transition_ckpt: bool = False,
):
    z = z_init + model.linear_z(model.layer_norm_z(z))
    z = z + model.template_embedder(
        batch=batch,
        z=z,
        pair_mask=pair_mask,
        chunk_size=mem.chunk_size,
        _mask_trans=True,
        use_deepspeed_evo_attention=False,
        use_cueq_triangle_kernels=mem.use_cueq_triangle_kernels,
        use_lma=mem.use_lma,
        inplace_safe=False,
    )
    m, msa_mask = model.msa_module_embedder(batch=batch, s_input=s_input)

    trans_ckpt = None
    if use_transition_ckpt:
        swiglu_cutoff = mem.msa_module.swiglu_chunk_token_cutoff
        trans_ckpt = (
            mem.msa_module.swiglu_seq_chunk_size
            if swiglu_cutoff is None or swiglu_cutoff > m.shape[-2]
            else None
        )

    z = model.msa_module(
        m,
        z,
        msa_mask=msa_mask.to(dtype=m.dtype),
        pair_mask=pair_mask.to(dtype=z.dtype),
        chunk_size=mem.chunk_size,
        transition_ckpt_chunk_size=trans_ckpt,
        use_deepspeed_evo_attention=False,
        use_cueq_triangle_kernels=mem.use_cueq_triangle_kernels,
        use_lma=mem.use_lma,
        inplace_safe=False,
        _mask_trans=True,
    )
    del m, msa_mask
    s = s_init + model.linear_s(model.layer_norm_s(s))
    return s, z


def _run_cycle(model, batch, s_input, s_init, z_init, s, z, token_mask, pair_mask, mem):
    s, z = _run_pre_pairformer(
        model,
        batch,
        s_input,
        s_init,
        z_init,
        s,
        z,
        pair_mask,
        mem,
        use_transition_ckpt=True,
    )
    s, z = model.pairformer_stack(
        s=s,
        z=z,
        single_mask=token_mask.to(dtype=z.dtype),
        pair_mask=pair_mask.to(dtype=s.dtype),
        chunk_size=mem.chunk_size,
        use_deepspeed_evo_attention=False,
        use_cueq_triangle_kernels=mem.use_cueq_triangle_kernels,
        use_lma=mem.use_lma,
        inplace_safe=False,
        _mask_trans=True,
    )
    return s, z


def _compute_confidence_scores(
    model: nn.Module,
    batch: dict,
    atom_xyz: torch.Tensor,
    si_trunk: torch.Tensor,
    zij_trunk: torch.Tensor,
    si_input: torch.Tensor,
):
    # Create dict for aux heads
    outputs = {
        "atom_positions_predicted": atom_xyz,
        "si_trunk": si_trunk,
        "zij_trunk": zij_trunk,
        "si_input": si_input,
    }

    # Run confidence heads
    confidence_metrics = model.aux_heads(
        batch=batch,
        si_input=outputs["si_input"],
        output=outputs,
        use_zij_trunk_embedding=True,
    )

    # Aggregate scores
    outputs.update(confidence_metrics)
    confidence_scores = get_confidence_scores(
        batch=batch,
        outputs=outputs,
        config=model.config,
    )

    return confidence_scores
