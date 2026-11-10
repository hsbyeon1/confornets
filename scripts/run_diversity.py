"""Benchmark k-confornet diversity training."""

import argparse
import logging
from pathlib import Path

from confornet.core.of3 import load_model
from confornet.data import Benchmark, PredictionQuery, load_test_cases
from confornet.inference.diversity import run_diversity_seed
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
    parser.add_argument(
        "--objective", choices=["coord_mse", "dist_cdf_mse"], default="coord_mse"
    )
    parser.add_argument(
        "--loss-res",
        type=str,
        default=None,
        help="Residue ranges the coord_mse diversity loss is computed over, e.g. "
        "'245-252' (res_id numbering of the query). Default: all atoms.",
    )
    parser.add_argument(
        "--loss-frame-res",
        type=str,
        default=None,
        help="Residue ranges whose CA atoms define the per-pair superposition, e.g. "
        "'3-24,43-64,122-139'. Default: all atoms.",
    )
    parser.add_argument(
        "--loss-atoms",
        choices=["ca", "heavy"],
        default="ca",
        help="Atoms of the --loss-res residues used in the loss.",
    )
    parser.add_argument(
        "--loss-chain",
        type=str,
        default="A",
        help="Chain ID the residue ranges refer to.",
    )
    parser.add_argument("--k-confornets", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=21)
    parser.add_argument("--num-runs", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=200,
        help="Diffusion steps for sample rollouts.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--save-steps",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20],
        help="Training steps at which confornets + samples are written.",
    )
    add_confidence_args(parser)
    args = parser.parse_args()
    resolve_confidence_args(args)

    local_rank, world_size, device = setup_device()

    benchmark = Benchmark.load(args.benchmark, Path(args.assets_dir))
    test_cases = load_test_cases(
        benchmark, args.test_case, args.num_nodes, args.node_idx
    )

    model = load_model(args.checkpoint, args.config_yaml, device)
    logger.info(f"Model loaded, c_z={model.config.architecture.shared.c_z}")

    jobs = []
    for tc in test_cases:
        qids = tc.query_ids
        base = args.num_runs // len(qids)
        rem = args.num_runs % len(qids)
        runs_per_query = [base + (1 if i < rem else 0) for i in range(len(qids))]
        for qid, n in zip(qids, runs_per_query):
            for seed in range(n):
                jobs.append((tc, qid, seed))
    logger.info(f"Total jobs: {len(jobs)}")

    query_cache: dict[str, PredictionQuery] = {}
    for job_idx, (tc, qid, seed) in enumerate(jobs):
        if job_idx % world_size != local_rank:
            continue
        if qid not in query_cache:
            qr = tc.query_ranges.get(qid, {})
            query_cache[qid] = PredictionQuery.from_batch_file(
                benchmark.get_batch_path(qid),
                device,
                alignment_resid_ranges=qr.get("alignment_resid_ranges"),
                metric_resid_ranges=qr.get("metric_resid_ranges"),
            )
        out_dir = Path(args.output_dir) / args.benchmark / tc.name
        logger.info(f"Rank {local_rank}: {tc.name}/{qid} seed {seed}")
        run_diversity_seed(
            model,
            query_cache[qid],
            qid,
            out_dir,
            seed,
            args,
            device,
            save_steps=args.save_steps,
        )

    logger.info("Done")


if __name__ == "__main__":
    main()
