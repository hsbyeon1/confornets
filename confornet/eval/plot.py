"""
Plotting utilities for evaluation results.

Mirrors bioemu_benchmarks' plot.py style for 2D free energy landscapes
and success rate curves.
"""

import copy
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import binned_statistic_2d

from confornet.eval.metrics import (
    adaptive_success_rate_at_pcts,
    success_rate_at_thresholds,
)

if TYPE_CHECKING:
    from confornet.eval.results import EvalResults, TestCaseResults

DEFAULT_FONTSIZE = 12
matplotlib.rcParams.update({"font.size": DEFAULT_FONTSIZE})
matplotlib.rc("xtick", labelsize=DEFAULT_FONTSIZE)
matplotlib.rc("ytick", labelsize=DEFAULT_FONTSIZE)


def plot_free_energy(
    x: np.ndarray,
    numbins: int = 20,
    max_energy: float = 10,
    levels: int = 20,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    cbar: bool = True,
    kT: float = 1,
    ax: Axes | None = None,
) -> Axes:
    """
    Plot 2D free energy landscape from RMSD coordinates.

    Args:
        x: Array of shape [N, 2] with RMSD values to two references
        numbins: Number of bins for histogram
        max_energy: Maximum energy for colormap capping
        levels: Number of contour levels
        x_range: (min, max) for x-axis, auto-computed if None
        y_range: (min, max) for y-axis, auto-computed if None
        cbar: Whether to plot colorbar
        kT: Temperature factor for energy
        ax: Matplotlib axes to plot on

    Returns:
        Axes with the plot
    """
    # Filter out NaN/inf rows
    mask = np.isfinite(x).all(axis=1)
    x = x[mask]

    if x_range is None:
        x_range = (min(x[:, 0]), max(x[:, 0]))
    if y_range is None:
        y_range = (min(x[:, 1]), max(x[:, 1]))

    # Shape x,y for plot
    grid_x = np.linspace(x_range[0], x_range[1], numbins)
    grid_x = np.stack([grid_x for _ in range(numbins)])
    grid_y = np.transpose(grid_x)

    # Bin the data
    binned = binned_statistic_2d(
        x[:, 0],
        x[:, 1],
        None,
        "count",
        bins=[numbins, numbins],
        range=[x_range, y_range],
    )
    grid_z = binned.statistic

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.subplots(1, 1)

    # Compute free energy
    energy = -np.log(grid_z + 1e-6)
    energy -= energy.min()
    energy = kT * energy
    energy = np.minimum(energy, max_energy + 1)

    # Colormap with white overflow
    cmap = copy.copy(plt.cm.turbo)
    cmap.set_over(color="w")

    # Generate contour plot
    ax.contourf(
        grid_x, grid_y, energy, cmap=cmap, levels=levels, vmin=0, vmax=max_energy
    )
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    if cbar:
        cbar_obj = plt.colorbar(ax.collections[0], ax=ax)
        cbar_obj.ax.set_ylim(0, max_energy)
        cbar_obj.set_label("Energy (kT)")

    return ax


def plot_free_energy_with_threshold(
    x: np.ndarray,
    success_threshold: float,
    max_range: float,
    tick_spacing: int | None = 2,
    numbins: int = 20,
    max_energy: float = 10,
    levels: int = 20,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    cbar: bool = False,
    kT: float = 1,
    line_color: str = "black",
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """
    Plot 2D free energy landscape with success threshold lines.

    Args:
        x: Array of shape [N, 2] with RMSD values to two references
        success_threshold: RMSD threshold for success (dashed lines)
        max_range: Maximum range for both axes
        tick_spacing: Spacing for ticks, None to skip
        numbins: Number of bins for histogram
        max_energy: Maximum energy for colormap capping
        levels: Number of contour levels
        x_range: (min, max) for x-axis, defaults to (0, max_range)
        y_range: (min, max) for y-axis, defaults to (0, max_range)
        cbar: Whether to plot colorbar
        kT: Temperature factor for energy
        line_color: Color of threshold lines
        ax: Matplotlib axes to plot on
        title: Plot title

    Returns:
        Axes with the plot
    """
    if x_range is None:
        x_range = (0, max_range)
    if y_range is None:
        y_range = (0, max_range)

    ax = plot_free_energy(
        x,
        numbins=numbins,
        max_energy=max_energy,
        levels=levels,
        x_range=x_range,
        y_range=y_range,
        cbar=cbar,
        kT=kT,
        ax=ax,
    )

    # Add threshold lines
    ax.axhline(y=success_threshold, color=line_color, linestyle="--")
    ax.axvline(x=success_threshold, color=line_color, linestyle="--")

    if tick_spacing is not None:
        ax.set_xticks(np.arange(0, max_range + 0.1, tick_spacing))
        ax.set_yticks(np.arange(0, max_range + 0.1, tick_spacing))

    if title:
        ax.set_title(title)

    return ax


def plot_2d_free_energy_grid(
    results: "EvalResults",
    max_range_multiplier: float = 2.0,
    numbins: int = 50,
    max_energy: float = 10,
    ncols: int = 3,
) -> Figure:
    """
    Create a grid of 2D free energy plots for all test cases.

    Args:
        results: EvalResults object
        max_range_multiplier: Multiplier for auto-computed max range
        numbins: Number of bins for histogram
        max_energy: Maximum energy for colormap
        ncols: Number of columns in the grid

    Returns:
        Figure with all plots
    """
    test_cases = list(results.test_cases.values())
    test_cases.sort(key=lambda tc: tc.test_case)
    n_plots = len(test_cases)
    nrows = int(np.ceil(n_plots / ncols))

    # Square subplots with 1:1 aspect ratio
    subplot_size = 3.5
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * subplot_size, nrows * subplot_size)
    )
    axes = np.atleast_2d(axes).T.flatten()

    threshold = results.threshold

    for i, tc_results in enumerate(test_cases):
        ax = axes[i]
        if tc_results.rmsd.shape[1] >= 2:
            x = tc_results.rmsd[:, :2]
            # Compute max_range based on minimum distances
            d = np.maximum(
                x[np.argmin(x[:, 0]), 1],
                x[np.argmin(x[:, 1]), 0],
            )
            max_range = max_range_multiplier * d
            if not np.isfinite(max_range) or max_range < 1:
                max_range = 1

            plot_free_energy_with_threshold(
                x,
                threshold,
                max_range=max_range,
                numbins=numbins,
                max_energy=max_energy,
                levels=20,
                ax=ax,
                line_color="red",
                tick_spacing=None,
                title=tc_results.test_case,
            )
            ax.set_aspect("equal")
        else:
            ax.text(
                0.5,
                0.5,
                "< 2 references",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(tc_results.test_case)

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    return fig


def plot_single_free_energy(
    tc_results: "TestCaseResults",
    threshold: float,
    max_range: float | None = None,
    max_range_multiplier: float = 2.0,
    numbins: int = 50,
    max_energy: float = 10,
) -> Figure:
    """
    Create a single 2D free energy plot for one test case.

    Args:
        tc_results: TestCaseResults for one test case
        threshold: RMSD threshold
        max_range: Maximum range for axes, auto-computed if None
        max_range_multiplier: Multiplier for auto-computed max range
        numbins: Number of bins
        max_energy: Maximum energy

    Returns:
        Figure with the plot
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    if tc_results.rmsd.shape[1] >= 2:
        x = tc_results.rmsd[:, :2]
        if max_range is None:
            d = np.maximum(
                x[np.argmin(x[:, 0]), 1],
                x[np.argmin(x[:, 1]), 0],
            )
            max_range = max_range_multiplier * d
            if not np.isfinite(max_range) or max_range < 1:
                max_range = 1

        plot_free_energy_with_threshold(
            x,
            threshold,
            max_range=max_range,
            numbins=numbins,
            max_energy=max_energy,
            levels=20,
            ax=ax,
            line_color="black",
            cbar=True,
            title=tc_results.test_case,
        )
        # Add reference names as axis labels
        if len(tc_results.references) >= 2:
            ax.set_xlabel(f"RMSD to {tc_results.references[0]} (Å)")
            ax.set_ylabel(f"RMSD to {tc_results.references[1]} (Å)")
    else:
        ax.text(
            0.5, 0.5, "< 2 references", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title(tc_results.test_case)

    fig.tight_layout()
    return fig


def plot_success_rate_vs_threshold(
    results: "EvalResults",
    thresholds: np.ndarray,
    k_values: list[int],
    n_bootstrap: int = 100,
    ax: Axes | None = None,
    benchmark_threshold: float | None = None,
) -> Axes:
    """
    Plot success rate vs RMSD threshold for different k values.

    Args:
        results: EvalResults object
        thresholds: Array of RMSD thresholds to sweep
        k_values: List of k values to plot curves for
        n_bootstrap: Number of bootstrap samples for uncertainty
        ax: Matplotlib axes
        benchmark_threshold: If provided, add vertical line at this threshold

    Returns:
        Axes with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(k_values)))

    for k, color in zip(k_values, colors):
        rates_dict = success_rate_at_thresholds(
            results, thresholds, k=k, n_bootstrap=n_bootstrap
        )

        means = np.array([rates_dict[t][0] for t in thresholds])
        stds = np.array([rates_dict[t][1] for t in thresholds])

        ax.plot(thresholds, means, label=f"k={k}", color=color, linewidth=2)
        ax.fill_between(thresholds, means - stds, means + stds, alpha=0.2, color=color)

    # Add benchmark threshold line if provided
    if benchmark_threshold is not None:
        ax.axvline(
            benchmark_threshold,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label=f"Benchmark threshold ({benchmark_threshold}Å)",
        )

    ax.set_xlabel("RMSD Threshold (Å)")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return ax


def plot_adaptive_success_rate(
    results: "EvalResults",
    pcts: np.ndarray,
    ref_to_ref_rmsd: dict[str, float],
    k_values: list[int],
    n_bootstrap: int = 100,
    ax: Axes | None = None,
) -> Axes:
    """Plot success rate vs % of ref-to-ref RMSD threshold."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(k_values)))

    for k, color in zip(k_values, colors):
        rates_dict = adaptive_success_rate_at_pcts(
            results, pcts, ref_to_ref_rmsd, k=k, n_bootstrap=n_bootstrap
        )

        means = np.array([rates_dict[float(p)][0] for p in pcts])
        stds = np.array([rates_dict[float(p)][1] for p in pcts])

        ax.plot(pcts * 100, means, label=f"k={k}", color=color, linewidth=2)
        ax.fill_between(pcts * 100, means - stds, means + stds, alpha=0.2, color=color)

    ax.set_xlabel("% of Ref-to-Ref RMSD")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(pcts[0] * 100, pcts[-1] * 100)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return ax
