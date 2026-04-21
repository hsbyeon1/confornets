"""Baseline OpenFold3 diffusion sampling (no confornets, no diversity training)."""

import argparse
import logging
from pathlib import Path

import torch

from confornet.core.of3 import load_model
from confornet.data import Benchmark, PredictionQuery, load_test_cases
from confornet.inference.transfer import sample_without_confornet
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


def main():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    add_benchmark_args(parser)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=200)
    add_confidence_args(parser)
    args = parser.parse_args()
    resolve_confidence_args(args)

    local_rank, world_size, device = setup_device()

    benchmark = Benchmark.load(args.benchmark, Path(args.assets_dir))
    test_cases = load_test_cases(
        benchmark, args.test_case, args.num_nodes, args.node_idx
    )

    model = load_model(args.checkpoint, args.config_yaml, device)
    model.eval()
    logger.info("Model loaded")

    # Build flat job list: (test_case, query_id, seed)
    jobs = []
    for tc in test_cases:
        query_ids = tc.query_ids
        base_seeds = args.num_seeds // len(query_ids)
        remainder = args.num_seeds % len(query_ids)
        seeds_per_query = [
            base_seeds + (1 if i < remainder else 0) for i in range(len(query_ids))
        ]
        for qid, n in zip(query_ids, seeds_per_query):
            for seed in range(n):
                jobs.append((tc, qid, seed))

    if len(jobs) % world_size != 0:
        logger.warning(
            f"Total jobs ({len(jobs)}) not divisible by world_size ({world_size}); "
            f"load will be uneven across ranks"
        )
    logger.info(f"Total jobs: {len(jobs)}")

    query_cache: dict[str, PredictionQuery] = {}
    for job_idx, (test_case, query_id, seed) in enumerate(jobs):
        if job_idx % world_size != local_rank:
            continue

        if query_id not in query_cache:
            batch_path = benchmark.get_batch_path(query_id)
            qr = test_case.query_ranges.get(query_id, {})
            query_cache[query_id] = PredictionQuery.from_batch_file(
                batch_path,
                device,
                alignment_resid_ranges=qr.get("alignment_resid_ranges"),
                metric_resid_ranges=qr.get("metric_resid_ranges"),
            )
        query = query_cache[query_id]

        logger.info(
            f"Rank {local_rank}, tc {test_case.name}, query {query_id}, seed {seed}"
        )

        torch.manual_seed(seed * 1000)
        samples, conf = sample_without_confornet(
            model,
            query,
            args.num_samples,
            device,
            args.num_steps,
            args.num_recycles,
            compute_confidence=args.compute_confidence,
        )

        # Output: {output_dir}/{benchmark}/{tc_name}/{query_id}/baseline/{tc_name}_seed_{seed}_sample_{i}_model.cif
        output_dir = (
            Path(args.output_dir)
            / args.benchmark
            / test_case.name
            / query_id
            / "baseline"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        file_prefix = f"{test_case.name}_seed_{seed}"
        for i in range(samples.shape[0]):
            cif_path = output_dir / f"{file_prefix}_sample_{i}_model.cif"
            write_cif(query.atom_array, samples[i], cif_path)
        save_confidence(
            conf,
            output_dir,
            file_prefix=file_prefix,
            save_full=args.save_full_confidence,
        )

        if conf is not None:
            avg_plddt = conf["plddt"].mean().item()
            gpde = conf["gpde"].mean().item()
            ptm = conf["ptm"].mean().item() if "ptm" in conf else float("nan")
            logger.info(f"  plddt={avg_plddt:.2f}, gpde={gpde:.4f}, ptm={ptm:.4f}")
        logger.info(f"  Saved {samples.shape[0]} samples to {output_dir}")

    logger.info("Done")


if __name__ == "__main__":
    main()
