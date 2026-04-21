import logging
from pathlib import Path

import torch
import torch.nn as nn
from openfold3.core.utils.tensor_utils import tensor_tree_map

from confornet.core.confornet import ConforNet
from confornet.core.diffusion import create_noise_schedule, diffusion_sample
from confornet.core.losses import aligned_mse_loss
from confornet.core.trunk import (
    ConforNetMode,
    _compute_confidence_scores,
    run_trunk_with_confornet,
)
from confornet.data import PredictionQuery, Reference
from confornet.inference.transfer import sample_with_confornet
from confornet.utils.align import get_matching_backbone_indices
from confornet.utils.confidence import save_confidence, save_training_confidence
from confornet.utils.csv_io import write_training_loss_csv
from confornet.utils.io import write_cif

logger = logging.getLogger(__name__)


def prepare_batch(query: PredictionQuery, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in query.batch.items()
    }


def mse_training_step(
    model: nn.Module,
    confornet: ConforNet,
    query: PredictionQuery,
    reference: Reference,
    query_idx: list[int],
    ref_idx: list[int],
    optimizer: torch.optim.Optimizer,
    noise_schedule: torch.Tensor,
    device: torch.device,
    num_recycles: int = 1,
    compute_confidence: bool = False,
) -> tuple[float, torch.Tensor, dict | None]:
    batch = prepare_batch(query, device)

    si_input, si_trunk, zij_trunk = run_trunk_with_confornet(
        model, batch, confornet, ConforNetMode.PAIR_PRE, num_recycles=num_recycles
    )

    si_input = si_input.unsqueeze(1)
    si_trunk = si_trunk.unsqueeze(1)
    zij_trunk = zij_trunk.unsqueeze(1)

    ref_space_uid_to_perm = batch.pop("ref_space_uid_to_perm", None)
    loss_weights = batch.pop("loss_weights", None)
    batch = tensor_tree_map(lambda t: t.unsqueeze(1) if t.dim() > 0 else t, batch)
    batch["ref_space_uid_to_perm"] = ref_space_uid_to_perm
    if loss_weights is not None:
        batch["loss_weights"] = loss_weights

    xl_pred = diffusion_sample(
        diffusion_module=model.diffusion_module,
        batch=batch,
        si_input=si_input,
        si_trunk=si_trunk,
        zij_trunk=zij_trunk,
        noise_schedule=noise_schedule,
        no_samples=1,
    )

    confidence_metrics = None
    if compute_confidence:
        with torch.no_grad():
            batch["atom_array"] = query.atom_array
            confidence_metrics = _compute_confidence_scores(
                model=model,
                batch=batch,
                atom_xyz=xl_pred,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                si_input=si_input,
            )
            confidence_metrics = {
                k: v.squeeze(0) for k, v in confidence_metrics.items()
            }

    # Extract backbone atoms and compute aligned MSE
    pred_bb = xl_pred[0, 0, query_idx]  # [N_bb, 3]
    ref_bb = reference.coordinates[ref_idx].to(device)  # [N_bb, 3]
    loss = aligned_mse_loss(pred_bb, ref_bb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), xl_pred.detach(), confidence_metrics


def run_mse_training(
    model: nn.Module,
    source_query: PredictionQuery,
    reference: Reference,
    output_dir: Path,
    args,
    device: torch.device,
    target_query: PredictionQuery | None = None,
) -> ConforNet:
    """Train one ConforNet on (source_query, reference), sample target_query (or source if None), write outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    c_z = model.config.architecture.shared.c_z
    noise_cfg = model.config.architecture.noise_schedule

    confornet = ConforNet(c_z).to(device)
    confornet.reset_to_identity()
    for p in confornet.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(confornet.parameters(), lr=args.lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
        if args.lr_annealing and args.max_steps > 0
        else None
    )

    noise_schedule = create_noise_schedule(
        no_rollout_steps=args.rollout_steps,
        sigma_data=noise_cfg.sigma_data,
        s_max=noise_cfg.s_max,
        s_min=noise_cfg.s_min,
        p=noise_cfg.p,
        dtype=torch.float32,
        device=device,
    )

    query_idx, ref_idx = get_matching_backbone_indices(
        source_query, reference, query_resid_ranges=source_query.alignment_resid_ranges
    )

    conf_rows = []
    loss_rows = []
    best_loss = float("inf")
    best_confornet_state = None
    consecutive_low = 0
    model.eval()
    for step in range(args.max_steps):
        loss, _, confidence_metrics = mse_training_step(
            model,
            confornet,
            source_query,
            reference,
            query_idx,
            ref_idx,
            optimizer,
            noise_schedule,
            device,
            args.num_recycles,
            compute_confidence=args.compute_confidence and step % args.log_every == 0,
        )
        if scheduler is not None:
            scheduler.step()

        if loss < best_loss:
            best_loss = loss
            best_confornet_state = {
                k: v.clone() for k, v in confornet.state_dict().items()
            }

        if loss < 0.1:
            consecutive_low += 1
            if consecutive_low >= 3:
                logger.info(f"Early stopping at step {step}: loss < 0.1 x 3")
                break
        else:
            consecutive_low = 0

        if step % args.log_every == 0:
            logger.info(f"Step {step}: loss={loss:.4f}")
        loss_rows.append((step, loss))

        if confidence_metrics is not None:
            row = [
                step,
                confidence_metrics["plddt"].mean().item(),
                confidence_metrics["gpde"].item(),
            ]
            if "ptm" in confidence_metrics:
                row.append(confidence_metrics["ptm"].item())
            conf_rows.append(tuple(row))

    write_training_loss_csv(output_dir / "training_loss.csv", loss_rows)
    save_training_confidence(output_dir / "training_confidence.csv", conf_rows)

    if best_confornet_state is not None:
        confornet.load_state_dict(best_confornet_state)
    torch.save(confornet.state_dict(), output_dir / "confornet.pt")

    sample_query = target_query or source_query
    samples, conf = sample_with_confornet(
        model,
        confornet,
        sample_query,
        args.num_samples,
        device,
        num_recycles=args.num_recycles,
        compute_confidence=args.compute_confidence,
    )
    for i in range(samples.shape[0]):
        write_cif(sample_query.atom_array, samples[i], output_dir / f"sample_{i}.cif")
    save_confidence(conf, output_dir, save_full=args.save_full_confidence)
    return confornet
