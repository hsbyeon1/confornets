import torch
import torch.nn as nn
from openfold3.core.utils.tensor_utils import tensor_tree_map

from confornet.core.confornet import ConforNet
from confornet.core.diffusion import create_noise_schedule
from confornet.core.trunk import (
    ConforNetMode,
    _compute_confidence_scores,
    run_trunk_with_confornet,
)
from confornet.data import PredictionQuery
from confornet.utils.confidence import stack_confidence_metrics


def sample_with_confornet(
    model: nn.Module,
    confornet: ConforNet | None,
    query: PredictionQuery,
    num_samples: int,
    device: torch.device,
    num_steps: int = 200,
    num_recycles: int = 10,
    compute_confidence: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in query.batch.items()
    }
    noise_cfg = model.config.architecture.noise_schedule

    noise_schedule = create_noise_schedule(
        no_rollout_steps=num_steps,
        sigma_data=noise_cfg.sigma_data,
        s_max=noise_cfg.s_max,
        s_min=noise_cfg.s_min,
        p=noise_cfg.p,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        si_input, si_trunk, zij_trunk = run_trunk_with_confornet(
            model, batch, confornet, ConforNetMode.PAIR_PRE, num_recycles
        )
        si_input = si_input.unsqueeze(1)
        si_trunk = si_trunk.unsqueeze(1)
        zij_trunk = zij_trunk.unsqueeze(1)

        ref_space_uid_to_perm = batch.pop("ref_space_uid_to_perm", None)
        batch = tensor_tree_map(lambda t: t.unsqueeze(1) if t.dim() > 0 else t, batch)
        batch["ref_space_uid_to_perm"] = ref_space_uid_to_perm
        batch["atom_array"] = [query.atom_array]

        coords_list, conf_list = [], []
        remaining = num_samples
        while remaining > 0:
            chunk = min(5, remaining)
            xl = model.sample_diffusion(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                noise_schedule=noise_schedule,
                no_rollout_samples=chunk,
            )
            coords_list.append(xl.squeeze(0))
            remaining -= chunk

            if compute_confidence:
                conf_list.append(
                    _compute_confidence_scores(
                        model=model,
                        batch=batch,
                        atom_xyz=xl,
                        si_trunk=si_trunk,
                        zij_trunk=zij_trunk,
                        si_input=si_input,
                    )
                )

    coords = torch.cat(coords_list, dim=0)[:num_samples]
    if compute_confidence:
        conf = stack_confidence_metrics(conf_list)
        conf = {k: v[:num_samples] for k, v in conf.items()}
    else:
        conf = None
    return coords, conf


def sample_without_confornet(
    model: nn.Module,
    query: PredictionQuery,
    num_samples: int,
    device: torch.device,
    num_steps: int = 200,
    num_recycles: int = 10,
    compute_confidence: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    return sample_with_confornet(
        model,
        None,
        query,
        num_samples,
        device,
        num_steps,
        num_recycles,
        compute_confidence=compute_confidence,
    )
