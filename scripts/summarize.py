"""Summarize and visualize evaluation results."""

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from confornet.data import load_config
from confornet.eval.metrics import benchmark_success_rate, load_ref_to_ref_rmsd
from confornet.eval.plot import (
    plot_2d_free_energy_grid,
    plot_adaptive_success_rate,
    plot_single_free_energy,
    plot_success_rate_vs_threshold,
)
from confornet.eval.results import EvalResults

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def iter_views(
    results: EvalResults, consistent: bool
) -> Iterator[tuple[str, EvalResults]]:
    """Yield (label, view) pairs: always overall (label=""); plus i/j for consistent benchmarks."""
    yield "", results
    if consistent:
        for ref_idx, label in [(0, "i"), (1, "j")]:
            view = results.filter_by_ref_index(ref_idx)
            if view.test_cases:
                yield label, view


def _format_success_rates(
    results: EvalResults,
    k_values: list[int],
    n_bootstrap: int,
    label: str = "overall",
) -> list[str]:
    """Render '@k=...: mean +/- std' lines for one results slice."""
    lines = [f"{label}:"]
    for k in k_values:
        mean, std = benchmark_success_rate(results, k=k, n_bootstrap=n_bootstrap)
        lines.append(f"  @k={k}: {mean:.3f} +/- {std:.3f}")
    return lines


def generate_summary_text(
    results: EvalResults,
    k_values: list[int],
    n_bootstrap: int = 100,
    consistent: bool = False,
) -> str:
    """Generate a text summary of evaluation results."""
    lines = [
        "Evaluation Summary",
        "==================",
        f"Benchmark: {results.benchmark}",
        f"Mode: {results.mode}",
        f"Threshold: {results.threshold} A",
        f"Test cases: {len(results.test_cases)}",
        "",
    ]

    total_samples = sum(len(tc.samples) for tc in results.test_cases.values())
    lines.append(f"Total samples: {total_samples}")
    lines.append("")

    lines.append("Success Rates")
    lines.append("-------------")
    for label, view in iter_views(results, consistent):
        section_label = "overall" if not label else f"{label}-only"
        lines.extend(_format_success_rates(view, k_values, n_bootstrap, section_label))
    lines.append("")

    lines.append("Per Test Case Summary")
    lines.append("---------------------")
    for tc_name, tc_results in sorted(results.test_cases.items()):
        min_rmsds = tc_results.get_min_rmsd()
        lines.append(
            f"  {tc_name}: {len(tc_results.samples)} samples, "
            f"min={min_rmsds.min():.2f}, mean={min_rmsds.mean():.2f}, max={min_rmsds.max():.2f}"
        )

    return "\n".join(lines)


def generate_transfer_aggregate_summary(
    source_results: dict[str, EvalResults],
    k_values: list[int],
    threshold: float,
    n_bootstrap: int = 100,
    consistent: bool = False,
) -> str:
    """Generate aggregate summary comparing all sources in transfer mode."""
    lines = [
        "Transfer Mode Aggregate Summary",
        "===============================",
        f"Threshold: {threshold} A",
        f"Sources: {len(source_results)}",
        "",
    ]

    def _table(table_label: str, ref_idx: int | None) -> list[str]:
        header = "  ".join([f"@k={k:2d}" for k in k_values])
        rows = [
            f"Success Rates by Source ({table_label})",
            "-" * (30 + len(header) + 2),
            f"{'Source':<30} {header}",
            "-" * (30 + len(header) + 2),
        ]
        for source_tc in sorted(source_results.keys()):
            r = source_results[source_tc]
            if ref_idx is not None:
                r = r.filter_by_ref_index(ref_idx)
                if not r.test_cases:
                    continue
            rates_str = [
                f"{benchmark_success_rate(r, k=k, n_bootstrap=n_bootstrap)[0]:.3f}"
                for k in k_values
            ]
            rows.append(f"{source_tc:<30} {'  '.join(rates_str)}")
        rows.append("")
        return rows

    lines.extend(_table("overall", None))
    if consistent:
        lines.extend(_table("i-only", 0))
        lines.extend(_table("j-only", 1))

    return "\n".join(lines)


def _plot_success_rate_views(
    results: EvalResults,
    output_dir: Path,
    thresholds: np.ndarray,
    k_values: list[int],
    n_bootstrap: int,
    consistent: bool,
):
    """Write success_rate_vs_rmsd plots for overall + (optionally) i/j views."""
    for label, view in iter_views(results, consistent):
        suffix = f"_{label}" if label else ""
        title_suffix = f" (conf {label})" if label else ""
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_success_rate_vs_threshold(
            view,
            thresholds,
            k_values,
            n_bootstrap=n_bootstrap,
            ax=ax,
            benchmark_threshold=results.threshold,
        )
        ax.set_title(
            f"Success Rate vs RMSD Threshold - {results.benchmark}{title_suffix}"
        )
        fig.savefig(
            output_dir / f"success_rate_vs_rmsd{suffix}.pdf", bbox_inches="tight"
        )
        plt.close(fig)


def _plot_adaptive_success_rate_views(
    results: EvalResults,
    output_dir: Path,
    pcts: np.ndarray,
    ref_to_ref_rmsd: dict[str, float],
    k_values: list[int],
    n_bootstrap: int,
    consistent: bool,
):
    """Write success_rate_vs_pct_rmsd plots for overall + (optionally) i/j views."""
    for label, view in iter_views(results, consistent):
        suffix = f"_{label}" if label else ""
        title_suffix = f" (conf {label})" if label else ""
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_adaptive_success_rate(
            view,
            pcts,
            ref_to_ref_rmsd,
            k_values,
            n_bootstrap=n_bootstrap,
            ax=ax,
        )
        ax.set_title(f"Adaptive Success Rate - {results.benchmark}{title_suffix}")
        fig.savefig(
            output_dir / f"success_rate_vs_pct_rmsd{suffix}.pdf", bbox_inches="tight"
        )
        plt.close(fig)


def summarize_standard(
    results: EvalResults,
    output_dir: Path,
    k_values: list[int],
    max_rmsd: float,
    threshold_steps: int,
    n_bootstrap: int,
    save_results_pkl: bool = True,
    individual_free_energy: bool = False,
    ref_to_ref_rmsd: dict[str, float] | None = None,
    consistent: bool = False,
):
    """Generate summary for standard (non-transfer) mode."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_results_pkl:
        results.save(output_dir / "results.pkl")
        logger.info(f"Saved results to {output_dir / 'results.pkl'}")

    summary_text = generate_summary_text(
        results, k_values, n_bootstrap, consistent=consistent
    )
    (output_dir / "summary.txt").write_text(summary_text)
    logger.info(f"Saved summary to {output_dir / 'summary.txt'}")
    print("\n" + summary_text)

    if len(results.test_cases) > 0:
        fig = plot_2d_free_energy_grid(results)
        fig.savefig(output_dir / "free_energy_combined.pdf", bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved combined free energy plot")

    if individual_free_energy:
        fe_dir = output_dir / "free_energy"
        fe_dir.mkdir(exist_ok=True)
        for tc_name, tc_results in results.test_cases.items():
            fig = plot_single_free_energy(tc_results, results.threshold)
            safe_name = tc_name.replace("/", "_")
            fig.savefig(fe_dir / f"{safe_name}.png", bbox_inches="tight", dpi=150)
            plt.close(fig)
        logger.info(f"Saved {len(results.test_cases)} individual free energy plots")

    thresholds = np.linspace(0.1, max_rmsd, threshold_steps)
    _plot_success_rate_views(
        results, output_dir, thresholds, k_values, n_bootstrap, consistent
    )
    logger.info("Saved success rate vs threshold plot(s)")

    if ref_to_ref_rmsd is not None:
        pcts = np.linspace(0.01, 0.5, 30)
        _plot_adaptive_success_rate_views(
            results,
            output_dir,
            pcts,
            ref_to_ref_rmsd,
            k_values,
            n_bootstrap,
            consistent,
        )
        logger.info("Saved adaptive success rate plot(s)")


def summarize_transfer(
    results: EvalResults,
    output_dir: Path,
    k_values: list[int],
    max_rmsd: float,
    threshold_steps: int,
    n_bootstrap: int,
    individual_free_energy: bool = False,
    ref_to_ref_rmsd: dict[str, float] | None = None,
    consistent: bool = False,
):
    """Generate summary for transfer mode with source grouping."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results.save(output_dir / "results.pkl")
    logger.info(f"Saved results to {output_dir / 'results.pkl'}")

    source_results = results.group_by_source()
    logger.info(f"Found {len(source_results)} source test cases")

    logger.info("Generating source_all (all sources as sampling pool)")
    summarize_standard(
        results,
        output_dir / "source_all",
        k_values,
        max_rmsd,
        threshold_steps,
        n_bootstrap,
        save_results_pkl=False,
        individual_free_energy=individual_free_energy,
        ref_to_ref_rmsd=ref_to_ref_rmsd,
        consistent=consistent,
    )

    for source_tc, src_results in source_results.items():
        safe_name = source_tc.replace("/", "_")
        summarize_standard(
            src_results,
            output_dir / f"source_{safe_name}",
            k_values,
            max_rmsd,
            threshold_steps,
            n_bootstrap,
            save_results_pkl=False,
            individual_free_energy=individual_free_energy,
            ref_to_ref_rmsd=ref_to_ref_rmsd,
            consistent=consistent,
        )

    source_results_with_all = {"all": results, **source_results}
    aggregate_summary = generate_transfer_aggregate_summary(
        source_results_with_all,
        k_values,
        results.threshold,
        n_bootstrap,
        consistent=consistent,
    )
    (output_dir / "summary.txt").write_text(aggregate_summary)
    logger.info(f"Saved aggregate transfer summary to {output_dir / 'summary.txt'}")
    print("\n" + aggregate_summary)


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="One or more .pkl files to merge/summarize",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for plots and summary",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="5,100",
        help="Comma-separated k values",
    )
    parser.add_argument(
        "--max-rmsd",
        type=float,
        default=6.0,
        help="Max RMSD for threshold sweep plot (default: 6.0)",
    )
    parser.add_argument(
        "--threshold-steps",
        type=int,
        default=30,
        help="Number of threshold steps (default: 30)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap samples (default: 100)",
    )
    parser.add_argument(
        "--individual-free-energy",
        action="store_true",
        help="Generate per-test-case free energy plots",
    )
    parser.add_argument(
        "--ref-to-ref-rmsd",
        type=str,
        default=None,
        help="Path to ref_to_ref_rmsd.csv",
    )
    args = parser.parse_args()

    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    input_paths = [Path(p) for p in args.input]
    logger.info(f"Loading {len(input_paths)} result files")
    results_list = [EvalResults.load(p) for p in input_paths]
    results = EvalResults.merge(results_list)
    logger.info(f"Merged results: {len(results.test_cases)} test cases")

    output_dir = Path(args.output)

    ref_to_ref_rmsd = None
    if args.ref_to_ref_rmsd:
        ref_to_ref_rmsd = load_ref_to_ref_rmsd(Path(args.ref_to_ref_rmsd))
        logger.info(f"Loaded ref-to-ref RMSD for {len(ref_to_ref_rmsd)} test cases")

    assets_dir = Path(results.assets_dir) if results.assets_dir else Path("./assets")
    config = load_config(assets_dir / "config.yaml")
    bm_config = config.get("benchmarks", {}).get(results.benchmark, {})
    consistent = bm_config.get(
        "consistent", config.get("defaults", {}).get("consistent", False)
    )

    if results.mode == "transfer":
        summarize_transfer(
            results,
            output_dir,
            k_values,
            args.max_rmsd,
            args.threshold_steps,
            args.n_bootstrap,
            individual_free_energy=args.individual_free_energy,
            ref_to_ref_rmsd=ref_to_ref_rmsd,
            consistent=consistent,
        )
    else:
        summarize_standard(
            results,
            output_dir,
            k_values,
            args.max_rmsd,
            args.threshold_steps,
            args.n_bootstrap,
            individual_free_energy=args.individual_free_energy,
            ref_to_ref_rmsd=ref_to_ref_rmsd,
            consistent=consistent,
        )

    logger.info(f"\nSummary complete. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
