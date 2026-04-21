"""Benchmark single-ConforNet MSE training toward reference structures."""

import argparse
import logging
from pathlib import Path

import torch

from confornet.core.of3 import load_model
from confornet.data import Benchmark, PredictionQuery, load_test_cases
from confornet.inference.mse_training import run_mse_training
from confornet.utils.cli import (
    add_benchmark_args,
    add_confidence_args,
    add_model_args,
    resolve_confidence_args,
    setup_device,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    add_benchmark_args(parser)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr-annealing", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--subsampled-all-msa-rows", type=int, default=None)
    add_confidence_args(parser)
    args = parser.parse_args()
    resolve_confidence_args(args)

    local_rank, world_size, device = setup_device()

    benchmark = Benchmark.load(args.benchmark, Path(args.assets_dir))
    test_cases = load_test_cases(
        benchmark, args.test_case, args.num_nodes, args.node_idx
    )

    model = load_model(
        args.checkpoint,
        args.config_yaml,
        device,
        subsampled_all_msa_rows=args.subsampled_all_msa_rows,
    )
    logger.info(f"Model loaded, c_z={model.config.architecture.shared.c_z}")

    jobs = [
        (tc, ref.name if len(tc.query_ids) > 1 else tc.query_ids[0], ref, run_idx)
        for tc in test_cases
        for ref in tc.references
        for run_idx in range(args.num_runs)
    ]
    logger.info(f"Total jobs: {len(jobs)}")

    query_cache: dict[str, PredictionQuery] = {}
    for job_idx, (tc, query_id, ref, run_idx) in enumerate(jobs):
        if job_idx % world_size != local_rank:
            continue
        if query_id not in query_cache:
            qr = tc.query_ranges.get(query_id, {})
            query_cache[query_id] = PredictionQuery.from_batch_file(
                benchmark.get_batch_path(query_id),
                device,
                alignment_resid_ranges=qr.get("alignment_resid_ranges"),
                metric_resid_ranges=qr.get("metric_resid_ranges"),
            )
        torch.manual_seed(run_idx * 1000)
        out_dir = (
            Path(args.output_dir)
            / args.benchmark
            / tc.name
            / ref.name
            / f"run_{run_idx}"
        )
        logger.info(
            f"{tc.name} ref {ref.name} query {query_id} run {run_idx} -> {out_dir}"
        )
        run_mse_training(
            model,
            query_cache[query_id],
            ref,
            out_dir,
            args,
            device,
        )

    logger.info("Done")


if __name__ == "__main__":
    main()
