"""Confidence metric helpers."""

from pathlib import Path

import torch

from confornet.utils.csv_io import fmt, write_csv


def stack_confidence_metrics(metrics_list: list[dict]) -> dict[str, torch.Tensor]:
    """Stack [1, batch, ...]-shaped confidence dicts into [total, ...] tensors. Non-tensor entries are skipped."""
    result = {}
    for k, v in metrics_list[0].items():
        if isinstance(v, torch.Tensor):
            result[k] = torch.cat([d[k].squeeze(0) for d in metrics_list], dim=0)
    return result


def save_confidence(
    conf: dict[str, torch.Tensor] | None,
    output_dir: Path,
    file_prefix: str = "",
    save_full: bool = False,
) -> None:
    """Write per-sample confidence.csv (plddt, gpde, ptm, iptm) and, if save_full, per-sample _confidence.pt tensors."""
    if conf is None:
        return

    prefix = f"{file_prefix}_" if file_prefix else ""
    n = conf["plddt"].shape[0]
    avg_plddt = conf["plddt"].mean(dim=-1)
    optional = [k for k in ("ptm", "iptm") if k in conf]

    rows = [
        (
            i,
            fmt(avg_plddt[i].item()),
            fmt(conf["gpde"][i].item()),
            *(fmt(conf[k][i].item()) for k in optional),
        )
        for i in range(n)
    ]
    write_csv(
        output_dir / f"{prefix}confidence.csv",
        ["sample", "plddt", "gpde", *optional],
        rows,
    )

    if save_full:
        for i in range(n):
            sample = {
                "plddt": conf["plddt"][i].detach().cpu(),
                "pde": conf["pde"][i].detach().cpu(),
            }
            for k in optional:
                sample[k] = conf[k][i].detach().cpu()
            torch.save(sample, output_dir / f"{prefix}sample_{i}_confidence.pt")


def save_training_confidence(path: Path, conf_rows: list[tuple]) -> None:
    """Write a per-step training-confidence log. Rows: (step, plddt, gpde[, ptm])."""
    if not conf_rows:
        return
    has_ptm = len(conf_rows[0]) > 3
    header = ["step", "plddt", "gpde"] + (["ptm"] if has_ptm else [])
    rows = [
        (r[0], fmt(r[1]), fmt(r[2]), *([fmt(r[3])] if has_ptm else []))
        for r in conf_rows
    ]
    write_csv(path, header, rows)
