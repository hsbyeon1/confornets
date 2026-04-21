"""Tiny CSV writer + shared training-log schema."""

import csv
from pathlib import Path


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    """Write `header` + `rows` to `path`. Rows should already contain
    primitive types or pre-formatted strings."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def write_training_loss_csv(
    path: Path,
    loss_rows: list[tuple],
    extra_cols: list[tuple[str, str]] = (),
) -> None:
    """Write a training-loss log. Each row is `(step, loss, *extras)`.

    `extra_cols` names and formats each trailing column, e.g.
    `[("pairwise_mse", ".6f"), ("lr", ".6e")]`. Loss is formatted as `.6f`.
    """
    header = ["step", "loss", *(c[0] for c in extra_cols)]
    rows = [
        (r[0], f"{r[1]:.6f}", *(format(v, s) for v, (_, s) in zip(r[2:], extra_cols)))
        for r in loss_rows
    ]
    write_csv(path, header, rows)
