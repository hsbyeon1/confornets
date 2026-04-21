"""Inference with trained ConforNet(s)."""

import argparse
import logging
from pathlib import Path

import torch

from confornet.data import Benchmark, PredictionQuery, load_test_cases
from confornet.core.confornet import ConforNet
from confornet.core.of3 import load_model
from confornet.inference.transfer import sample_with_confornet
from confornet.utils.cli import (
    add_benchmark_args,
    add_confidence_args,
    add_model_args,
    resolve_confidence_args,
    setup_device,
)
from confornet.utils.confidence import save_confidence
from confornet.utils.io import write_cif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_samples_per_query(num_samples: int, num_queries: int) -> list[int]:
    """Distribute samples among queries (extras go to first queries)."""
    base = num_samples // num_queries
    remainder = num_samples % num_queries
    return [base + (1 if i < remainder else 0) for i in range(num_queries)]


def discover_confornets(
    mse_dir: Path, benchmark: str, source_cases: list[str] | None
) -> list[tuple[Path, str]]:
    """
    Discover all confornet.pt files from MSE output directory.

    Args:
        mse_dir: Root MSE training output directory
        benchmark: Benchmark name
        source_cases: List of source test cases to include (None = all)

    Returns:
        List of (confornet_path, output_prefix) tuples sorted by path
    """
    benchmark_dir = mse_dir / benchmark

    if not benchmark_dir.exists():
        raise ValueError(f"Benchmark directory not found: {benchmark_dir}")

    if source_cases is None:
        source_cases = [d.name for d in benchmark_dir.iterdir() if d.is_dir()]

    confornets = []
    for tc in source_cases:
        tc_dir = benchmark_dir / tc
        if tc_dir.exists():
            for confornet_path in tc_dir.glob("*/run_*/confornet.pt"):
                run_dir = confornet_path.parent
                ref_dir = run_dir.parent
                run_idx = int(run_dir.name.split("_")[1])
                output_prefix = f"source_{tc}/ref_{ref_dir.name}_run_{run_idx}"
                confornets.append((confornet_path, output_prefix))

    return sorted(confornets, key=lambda x: str(x[0]))


def main():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    add_benchmark_args(parser)

    # Single ConforNet mode
    parser.add_argument(
        "--confornet-path",
        type=str,
        default=None,
        help="Path to single trained ConforNet (single mode)",
    )

    # Auto-discover mode
    parser.add_argument(
        "--mse-dir",
        type=str,
        default=None,
        help="MSE training output directory (auto-discovery mode)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Comma-separated source test cases (default: all in mse-dir)",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    add_confidence_args(parser)
    args = parser.parse_args()
    resolve_confidence_args(args)

    single_mode = args.confornet_path is not None
    discover_mode = args.mse_dir is not None

    if single_mode and discover_mode:
        parser.error("Cannot use both --confornet-path and --mse-dir")
    if not single_mode and not discover_mode:
        parser.error("Must specify either --confornet-path or --mse-dir")

    local_rank, world_size, device = setup_device()

    benchmark = Benchmark.load(args.benchmark, Path(args.assets_dir))
    target_test_cases = load_test_cases(
        benchmark, args.test_case, args.num_nodes, args.node_idx
    )

    logger.info(f"Target test cases: {[tc.name for tc in target_test_cases]}")

    if single_mode:
        confornets = [(Path(args.confornet_path), "")]
    else:
        mse_dir = Path(args.mse_dir)
        source_cases = args.source.split(",") if args.source else None
        confornets = discover_confornets(mse_dir, args.benchmark, source_cases)

        if len(confornets) == 0:
            logger.warning("No confornets found, exiting")
            return

    logger.info(f"ConforNets: {len(confornets)}")

    jobs = []
    for confornet_path, output_prefix in confornets:
        for target_tc in target_test_cases:
            query_ids = target_tc.query_ids
            samples_per_query = compute_samples_per_query(
                args.num_samples, len(query_ids)
            )
            for query_id, num_samples_for_query in zip(query_ids, samples_per_query):
                assert num_samples_for_query > 0
                jobs.append(
                    (
                        confornet_path,
                        output_prefix,
                        target_tc,
                        query_id,
                        num_samples_for_query,
                    )
                )

    logger.info(f"Total jobs: {len(jobs)}")

    model = load_model(args.checkpoint, args.config_yaml, device)
    c_z = model.config.architecture.shared.c_z
    logger.info(f"Model loaded, c_z={c_z}")

    query_cache: dict[str, PredictionQuery] = {}

    for job_idx, (
        confornet_path,
        output_prefix,
        target_tc,
        query_id,
        num_samples_for_query,
    ) in enumerate(jobs):
        if job_idx % world_size != local_rank:
            continue

        logger.info(
            f"Job {job_idx}: confornet={confornet_path.name}, target={target_tc.name}, query={query_id}, samples={num_samples_for_query}"
        )

        confornet = ConforNet(c_z).to(device)
        confornet.load_state_dict(torch.load(confornet_path, map_location=device))
        confornet.eval()

        if query_id not in query_cache:
            batch_path = benchmark.get_batch_path(query_id)
            qr = target_tc.query_ranges.get(query_id, {})
            query_cache[query_id] = PredictionQuery.from_batch_file(
                batch_path,
                device,
                alignment_resid_ranges=qr.get("alignment_resid_ranges"),
                metric_resid_ranges=qr.get("metric_resid_ranges"),
            )
        query = query_cache[query_id]

        samples, conf = sample_with_confornet(
            model,
            confornet,
            query,
            num_samples_for_query,
            device,
            200,  # full rollout
            args.num_recycles,
            compute_confidence=args.compute_confidence,
        )

        if output_prefix:
            output_subdir = (
                Path(args.output_dir)
                / args.benchmark
                / output_prefix.split("/")[0]
                / target_tc.name
                / query_id
            )
            file_prefix = output_prefix.split("/")[1]
        else:
            output_subdir = (
                Path(args.output_dir) / args.benchmark / target_tc.name / query_id
            )
            file_prefix = ""

        output_subdir.mkdir(parents=True, exist_ok=True)

        for i in range(samples.shape[0]):
            if file_prefix:
                cif_path = output_subdir / f"{file_prefix}_sample_{i}.cif"
            else:
                cif_path = output_subdir / f"sample_{i}.cif"
            write_cif(query.atom_array, samples[i], cif_path)

        logger.info(f"  Saved {num_samples_for_query} samples to {output_subdir}")

        save_confidence(conf, output_subdir, file_prefix, args.save_full_confidence)

    logger.info("Done")


if __name__ == "__main__":
    main()
