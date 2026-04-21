import copy
import logging
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
from openfold3.core.utils.tensor_utils import tensor_tree_map

from confornet.core.confornet import ConforNet
from confornet.core.diffusion import create_noise_schedule, diffusion_sample
from confornet.core.losses import (
    distogram_to_cdf,
    pairwise_mse_loss,
    pairwise_tensor_mse,
)
from confornet.core.trunk import (
    _compute_confidence_scores,
    run_batched_trunk_k_confornets,
)
from confornet.data import PredictionQuery
from confornet.utils.confidence import save_confidence, stack_confidence_metrics
from confornet.utils.csv_io import write_training_loss_csv
from confornet.utils.io import write_cif

logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Diversity objectives. `requires_diffusion` gates whether a rollout is needed."""

    COORD_MSE = "coord_mse"
    DIST_CDF_MSE = "dist_cdf_mse"

    @property
    def requires_diffusion(self) -> bool:
        return self is ObjectiveType.COORD_MSE

    @property
    def metric_key(self) -> str:
        return {
            ObjectiveType.COORD_MSE: "pairwise_mse",
            ObjectiveType.DIST_CDF_MSE: "pairwise_cdf_mse",
        }[self]


class KConforNetManager:
    def __init__(
        self,
        k: int,
        c_z: int,
        device: torch.device,
        lr: float,
        grad_clip: float = 10.0,
    ):
        self.k = k
        self.grad_clip = grad_clip
        self.confornets = []

        for _ in range(k):
            confornet = ConforNet(c_z).to(device)
            confornet.reset_to_identity()
            with torch.no_grad():
                confornet.W.add_(torch.randn_like(confornet.W) * 0.01)
                confornet.b.add_(torch.randn_like(confornet.b) * 0.01)
            for p in confornet.parameters():
                p.requires_grad_(True)
            self.confornets.append(confornet)

        all_params = [p for c in self.confornets for p in c.parameters()]
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        self.initial_lr = lr

    def step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        for confornet in self.confornets:
            torch.nn.utils.clip_grad_norm_(confornet.parameters(), self.grad_clip)
        self.optimizer.step()

    def halve_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 2.0

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def diversity_training_step(
    model: nn.Module,
    confornet_manager: KConforNetManager,
    query: PredictionQuery,
    noise_schedule: torch.Tensor | None,
    fixed_noise: torch.Tensor | None,
    device: torch.device,
    num_recycles: int = 10,
    objective: ObjectiveType = ObjectiveType.COORD_MSE,
) -> tuple[float, torch.Tensor | None, dict]:
    k = confornet_manager.k
    batch = query.batch

    si_input, si_trunk, zij_trunk = run_batched_trunk_k_confornets(
        model, batch, confornet_manager.confornets, num_recycles
    )

    if objective is ObjectiveType.COORD_MSE:
        if fixed_noise is None or noise_schedule is None:
            raise ValueError(
                "coord_mse objective requires fixed_noise and noise_schedule"
            )

        ref_space_uid_to_perm = batch.pop("ref_space_uid_to_perm", None)
        loss_weights = batch.pop("loss_weights", None)
        batch_exp = tensor_tree_map(
            lambda t: t.unsqueeze(1) if t.dim() > 0 else t, batch
        )

        def expand_for_k(t):
            if t.dim() == 0:
                return t
            return t.repeat(k, *([1] * (t.dim() - 1)))

        batch_k = tensor_tree_map(expand_for_k, batch_exp)
        batch_k["ref_space_uid_to_perm"] = ref_space_uid_to_perm
        if loss_weights is not None:
            batch_k["loss_weights"] = loss_weights

        fixed_noise_k = fixed_noise.repeat(k, 1, 1, 1)
        xl_pred = diffusion_sample(
            diffusion_module=model.diffusion_module,
            batch=batch_k,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            noise_schedule=noise_schedule,
            no_samples=1,
            initial_noise=fixed_noise_k,
        )
        loss = pairwise_mse_loss(xl_pred)
        xl_pred_detached = xl_pred.detach()
    else:  # DIST_CDF_MSE
        zij = zij_trunk.squeeze(1)  # [k, N, N, c_z]
        distogram_logits = model.aux_heads.distogram(z=zij)
        cdf = distogram_to_cdf(distogram_logits)
        loss = pairwise_tensor_mse(cdf)
        xl_pred_detached = None

    confornet_manager.step(loss)

    metrics = {
        "loss": loss.item(),
        objective.metric_key: -loss.item(),
        "lr": confornet_manager.get_lr(),
    }
    return loss.item(), xl_pred_detached, metrics


def generate_rollout_samples(
    model: nn.Module,
    confornet: ConforNet,
    query: PredictionQuery,
    device: torch.device,
    num_samples: int = 5,
    num_steps: int = 200,
    num_recycles: int = 10,
    compute_confidence: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    batch = query.batch
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
        si_input, si_trunk, zij_trunk = run_batched_trunk_k_confornets(
            model, batch, [confornet], num_recycles
        )

        ref_space_uid_to_perm = batch.get("ref_space_uid_to_perm", None)
        loss_weights = batch.get("loss_weights", None)
        batch_exp = tensor_tree_map(
            lambda t: t.unsqueeze(1)
            if isinstance(t, torch.Tensor) and t.dim() > 0
            else t,
            {
                k: v
                for k, v in batch.items()
                if k not in ["ref_space_uid_to_perm", "loss_weights"]
            },
        )
        batch_exp["ref_space_uid_to_perm"] = ref_space_uid_to_perm
        if loss_weights is not None:
            batch_exp["loss_weights"] = loss_weights
        batch_exp["atom_array"] = [query.atom_array]

        coords_list = []
        conf_list = []
        remaining = num_samples
        while remaining > 0:
            chunk = min(5, remaining)
            xl_pred = model.sample_diffusion(
                batch=batch_exp,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                noise_schedule=noise_schedule,
                no_rollout_samples=chunk,
            )
            coords_list.append(xl_pred.squeeze(0))
            remaining -= chunk

            if compute_confidence:
                conf_list.append(
                    _compute_confidence_scores(
                        model=model,
                        batch=batch_exp,
                        atom_xyz=xl_pred,
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


def run_diversity_seed(
    model: nn.Module,
    query: PredictionQuery,
    query_id: str,
    output_dir: Path,
    seed: int,
    args,
    device: torch.device,
    *,
    save_steps: list[int],
) -> None:
    """Run one diversity seed. `save_steps` must be non-empty; each step triggers a rollout+confornet save."""
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed * 1000)

    objective = ObjectiveType(args.objective)
    noise_cfg = model.config.architecture.noise_schedule
    c_z = model.config.architecture.shared.c_z
    manager = KConforNetManager(args.k_confornets, c_z, device, args.lr, args.grad_clip)

    noise_schedule = None
    fixed_noise = None
    if objective.requires_diffusion:
        noise_schedule = create_noise_schedule(
            no_rollout_steps=1,
            sigma_data=noise_cfg.sigma_data,
            s_max=noise_cfg.s_max,
            s_min=noise_cfg.s_min,
            p=noise_cfg.p,
            dtype=torch.bfloat16,
            device=device,
        )
        num_atoms = query.batch["atom_mask"].shape[-1]
        fixed_noise = torch.randn(
            (1, 1, num_atoms, 3), device=device, dtype=torch.bfloat16
        )

    loss_rows = []
    model.eval()
    for step in range(args.max_steps):
        batch_copy = copy.deepcopy(query.batch)
        q = PredictionQuery(
            query.atom_array,
            query.sequence,
            batch_copy,
            query.alignment_resid_ranges,
            query.metric_resid_ranges,
        )
        loss, _, m = diversity_training_step(
            model,
            manager,
            q,
            noise_schedule,
            fixed_noise,
            device,
            args.num_recycles,
            objective,
        )
        loss_rows.append((step, m["loss"], m[objective.metric_key], m["lr"]))

        if step % args.log_every == 0:
            logger.info(
                f"Seed {seed}, step {step}: loss={loss:.4f} "
                f"{objective.metric_key}={m[objective.metric_key]:.4f}"
            )

        if step > 0 and step % 5 == 0:
            manager.halve_lr()

        if step in save_steps:
            for i, confornet in enumerate(manager.confornets):
                samples, conf = generate_rollout_samples(
                    model,
                    confornet,
                    query,
                    device,
                    args.num_samples,
                    args.num_steps,
                    args.num_recycles,
                    compute_confidence=args.compute_confidence,
                )
                prefix = f"{query_id}_step_{step}_confornet_{i}"
                for j in range(samples.shape[0]):
                    write_cif(
                        query.atom_array,
                        samples[j],
                        seed_dir / f"{prefix}_sample_{j}.cif",
                    )
                save_confidence(
                    conf,
                    seed_dir,
                    file_prefix=prefix,
                    save_full=args.save_full_confidence,
                )
                torch.save(confornet.state_dict(), seed_dir / f"{prefix}.pt")

    write_training_loss_csv(
        seed_dir / "training_loss.csv",
        loss_rows,
        extra_cols=[(objective.metric_key, ".6f"), ("lr", ".6e")],
    )
    (seed_dir / "objective.txt").write_text(objective.value + "\n")
