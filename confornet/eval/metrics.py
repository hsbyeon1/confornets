"""Bioemu-style success-rate metrics: outer-bootstrap loop, direct `np.std` over trials."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from confornet.eval.results import EvalResults


def success_at_k(rmsds: np.ndarray, threshold: float, k: int = 1) -> bool:
    """
    Success if any of k random samples is below threshold.

    Args:
        rmsds: Array of RMSD values, shape [n_samples]
        threshold: RMSD threshold for success
        k: Number of samples to select

    Returns:
        True if any of the k selected samples is below threshold
    """
    n = len(rmsds)
    if n == 0:
        return False
    if k > n:
        k = n
    selected = np.random.choice(rmsds, size=k, replace=False)
    return bool(np.any(selected < threshold))


def _benchmark_success(
    results: "EvalResults",
    threshold_fn,
    k: int,
    n_bootstrap: int,
) -> tuple[float, float]:
    """Outer bootstrap over `n_bootstrap` trials. `threshold_fn(tc) -> float | None` picks
    the per-TC threshold (None skips the TC). Each trial: `success_at_k` per (tc, ref) →
    mean across refs (per tc) → mean across tcs (overall). Returns (mean, std) of the
    trial-level overall success rate."""
    trials = []
    for _ in range(n_bootstrap):
        tc_means = []
        for tc, tc_results in results.test_cases.items():
            thr = threshold_fn(tc)
            if thr is None:
                continue
            hits = [
                float(success_at_k(tc_results.rmsd[:, j], thr, k))
                for j in range(len(tc_results.references))
            ]
            if hits:
                tc_means.append(float(np.mean(hits)))
        if tc_means:
            trials.append(float(np.mean(tc_means)))
    if not trials:
        return (0.0, 0.0)
    return (float(np.mean(trials)), float(np.std(trials)))


def benchmark_success_rate(
    results: "EvalResults",
    k: int = 1,
    n_bootstrap: int = 100,
) -> tuple[float, float]:
    """Overall success rate at the benchmark's default threshold."""
    return _benchmark_success(results, lambda _tc: results.threshold, k, n_bootstrap)


def success_rate_at_thresholds(
    results: "EvalResults",
    thresholds: np.ndarray,
    k: int = 1,
    n_bootstrap: int = 100,
) -> dict[float, tuple[float, float]]:
    """Sweep of overall success rate over RMSD thresholds."""
    return {
        float(t): _benchmark_success(results, lambda _tc, t=t: float(t), k, n_bootstrap)
        for t in thresholds
    }


def load_ref_to_ref_rmsd(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    return dict(zip(df["test_case"], df["rmsd"]))


def adaptive_benchmark_success_rate(
    results: "EvalResults",
    pct: float,
    ref_to_ref_rmsd: dict[str, float],
    k: int = 1,
    n_bootstrap: int = 100,
) -> tuple[float, float]:
    """Per-TC threshold = `pct * ref_to_ref_rmsd[tc]`; TCs not in `ref_to_ref_rmsd` are skipped."""
    return _benchmark_success(
        results,
        lambda tc: pct * ref_to_ref_rmsd[tc] if tc in ref_to_ref_rmsd else None,
        k,
        n_bootstrap,
    )


def adaptive_success_rate_at_pcts(
    results: "EvalResults",
    pcts: np.ndarray,
    ref_to_ref_rmsd: dict[str, float],
    k: int = 1,
    n_bootstrap: int = 100,
) -> dict[float, tuple[float, float]]:
    """Sweep of adaptive success rate over pct scalings."""
    return {
        float(p): adaptive_benchmark_success_rate(
            results, p, ref_to_ref_rmsd, k, n_bootstrap
        )
        for p in pcts
    }
