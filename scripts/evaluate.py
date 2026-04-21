"""
Evaluate predicted structures against references.
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from confornet.data import Benchmark, PredictionQuery, load_config, load_test_cases
from confornet.eval.results import EvalResults, TestCaseResults
from confornet.utils.filepath_parsers import (
    parse_diversity_path,
    parse_mse_path,
    parse_of3p_path,
    parse_transfer_path,
)
from confornet.utils.io import load_structure
from confornet.utils.usalign import compute_rmsd_usalign

# USAlign binary path (relative to repo root)
USALIGN_PATH = Path(__file__).parent.parent / "packages" / "USalign"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def compute_sample_metrics(
    cif_path: Path,
    query: PredictionQuery,
    references: list,
    usalign_path: str,
) -> np.ndarray:
    """Compute RMSD for one sample against all references."""
    sample_struct = load_structure(cif_path)
    coords = torch.from_numpy(sample_struct.coord).float()

    rmsds = np.zeros(len(references))

    for j, ref in enumerate(references):
        rmsds[j] = compute_rmsd_usalign(coords, query, ref, usalign_path)

    return rmsds


def discover_structure_files(
    samples_dir: Path,
    test_case,
    mode: str,
) -> tuple[list[Path], list]:
    """Discover structure files (.cif/.pdb) and parse metadata based on mode."""
    tc_name = test_case.name

    if mode == "transfer":
        glob_patterns = [
            f"source_*/{tc_name}/**/*.cif",
            f"source_*/{tc_name}/**/*.pdb",
            f"{tc_name}/**/*.cif",
            f"{tc_name}/**/*.pdb",
        ]
    else:
        glob_patterns = [
            f"{qid}/**/*.{ext}" for qid in test_case.query_ids for ext in ("cif", "pdb")
        ]

    cif_files = sorted(p for pat in glob_patterns for p in samples_dir.glob(pat))

    if mode == "mse":
        # Build ref_name -> query_id mapping.
        # 1-query: broadcast the lone query to all refs. 2-query: query filename == ref.name.
        if len(test_case.query_ids) == 1:
            ref_to_query = {
                ref.name: test_case.query_ids[0] for ref in test_case.references
            }
        else:
            ref_to_query = {ref.name: ref.name for ref in test_case.references}

    metadata_list = []

    for cif_path in cif_files:
        if mode == "diversity":
            meta = parse_diversity_path(cif_path, samples_dir)
        elif mode == "mse":
            meta = parse_mse_path(cif_path, samples_dir, tc_name, ref_to_query)
        elif mode == "transfer":
            meta = parse_transfer_path(cif_path, samples_dir)
        elif mode == "of3p":
            meta = parse_of3p_path(cif_path, samples_dir)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if meta is not None:
            metadata_list.append(meta)

    valid_cifs = [m.path for m in metadata_list]
    return valid_cifs, metadata_list


def evaluate_test_case(
    test_case,
    cif_files: list[Path],
    metadata_list: list,
    queries: dict[str, PredictionQuery],
    mode: str,
    n_jobs: int = -1,
    executor_type: str = "process",
) -> TestCaseResults:
    """
    Evaluate all samples for a test case using multiprocessing.

    Args:
        test_case: TestCase object with references
        cif_files: List of paths to predicted structure CIFs
        metadata_list: List of sample metadata
        queries: Dict mapping query_id to PredictionQuery
        mode: Evaluation mode (diversity, mse, transfer, of3p)
        n_jobs: Number of parallel jobs (-1 for all CPUs)
    """
    if len(cif_files) == 0:
        # Return empty results
        return TestCaseResults(
            test_case=test_case.name,
            references=[ref.name for ref in test_case.references],
            samples=[],
            rmsd=np.array([]).reshape(0, len(test_case.references)),
        )

    usalign_path = str(USALIGN_PATH)
    if not USALIGN_PATH.exists():
        raise RuntimeError(f"USAlign binary not found at {USALIGN_PATH}")

    # Get query for each sample
    def get_query_for_sample(meta):
        return queries[meta.target_query_id]

    # Determine number of workers
    if n_jobs == -1:
        n_workers = min(16, os.cpu_count() or 1)
    elif n_jobs < 0:
        n_workers = min(16, max(1, (os.cpu_count() or 1) + 1 + n_jobs))
    else:
        n_workers = n_jobs

    Executor = ProcessPoolExecutor if executor_type == "process" else ThreadPoolExecutor
    results = [None] * len(cif_files)
    batch_size = n_workers * 4

    with Executor(max_workers=n_workers) as executor:
        with tqdm(total=len(cif_files), desc="Computing metrics") as pbar:
            for batch_start in range(0, len(cif_files), batch_size):
                batch_end = min(batch_start + batch_size, len(cif_files))
                future_to_idx = {
                    executor.submit(
                        compute_sample_metrics,
                        cif_files[i],
                        get_query_for_sample(metadata_list[i]),
                        test_case.references,
                        usalign_path,
                    ): i
                    for i in range(batch_start, batch_end)
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    pbar.update(1)

    # Stack into arrays
    rmsd_array = np.stack([r for r in results])  # [N, R]

    return TestCaseResults(
        test_case=test_case.name,
        references=[ref.name for ref in test_case.references],
        samples=metadata_list,
        rmsd=rmsd_array,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted structures")
    parser.add_argument("--samples-dir", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument(
        "--mode",
        choices=["diversity", "mse", "transfer", "of3p"],
        required=True,
    )
    parser.add_argument("--assets-dir", type=str, default="./assets")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--test-cases",
        type=str,
        default=None,
        help="Comma-separated list of test cases to evaluate",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of parallel jobs"
    )
    parser.add_argument(
        "--executor",
        choices=["process", "thread"],
        default="process",
        help="Executor type for parallel evaluation",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Total number of nodes for distributed evaluation",
    )
    parser.add_argument(
        "--node-idx",
        type=int,
        default=0,
        help="Index of this node (0-indexed)",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    assets_dir = Path(args.assets_dir)
    output_path = Path(args.output)

    # Load benchmark and config
    benchmark = Benchmark.load(args.benchmark, assets_dir)
    config = load_config(assets_dir / "config.yaml")
    threshold = (
        config.get("benchmarks", {})
        .get(args.benchmark, {})
        .get("rmsd_threshold", config["defaults"]["rmsd_threshold"])
    )

    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"RMSD threshold: {threshold}")
    logger.info(f"Samples dir: {samples_dir}")

    # Filter and chunk test cases
    test_cases = load_test_cases(
        benchmark, args.test_cases, args.num_nodes, args.node_idx
    )

    logger.info(f"Evaluating {len(test_cases)} test cases")

    # Evaluate each test case
    all_results: dict[str, TestCaseResults] = {}

    for tc in (pbar := tqdm(test_cases, desc="Test cases")):
        pbar.set_postfix_str(tc.name)

        # Discover CIF files
        cif_files, metadata_list = discover_structure_files(samples_dir, tc, args.mode)

        if len(cif_files) == 0:
            tqdm.write(f"  No CIF files found for {tc.name}, skipping")
            continue

        # Load queries for each unique target_query_id in metadata
        unique_query_ids = set(m.target_query_id for m in metadata_list)
        queries = {}
        for qid in unique_query_ids:
            batch_path = benchmark.get_batch_path(qid)
            qr = tc.query_ranges.get(qid, {})
            queries[qid] = PredictionQuery.from_batch_file(
                batch_path,
                alignment_resid_ranges=qr.get("alignment_resid_ranges"),
                metric_resid_ranges=qr.get("metric_resid_ranges"),
            )

        # Evaluate
        tc_results = evaluate_test_case(
            tc,
            cif_files,
            metadata_list,
            queries,
            mode=args.mode,
            n_jobs=args.n_jobs,
            executor_type=args.executor,
        )
        all_results[tc.name] = tc_results

        # Print summary
        min_rmsds = tc_results.get_min_rmsd()
        tqdm.write(
            f"  {tc.name}: Min RMSD: {min_rmsds.min():.2f}, "
            f"Mean: {min_rmsds.mean():.2f}, Max: {min_rmsds.max():.2f}"
        )

    # Build EvalResults
    eval_results = EvalResults(
        benchmark=args.benchmark,
        mode=args.mode,
        threshold=threshold,
        test_cases=all_results,
        samples_dir=str(samples_dir),
        assets_dir=str(assets_dir),
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_results.save(output_path)
    logger.info(f"\nSaved results to {output_path}")

    # Print overall summary
    logger.info("\n=== Overall Summary ===")
    logger.info(eval_results.summary())

    # Compute and print success rates
    mean, std = eval_results.compute_success_rates(k=1, n_bootstrap=100)
    logger.info(f"\nSuccess rate @1: {mean:.3f} +/- {std:.3f}")


if __name__ == "__main__":
    main()
