"""Preprocessing pipeline for confornet benchmarks.

Usage:
    python preprocess.py --benchmark domainmotion --assets-dir /path/to/assets
    python preprocess.py --benchmark domainmotion --assets-dir /path/to/assets --skip-msa
"""

import argparse
import json
import shutil
from pathlib import Path

from confornet.preprocess import (
    load_query_definitions,
    run_msa_batched,
    save_batches,
)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline for confornet benchmarks"
    )
    parser.add_argument("--assets-dir", type=Path, default=Path("./assets"))
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--skip-msa", action="store_true")
    args = parser.parse_args()

    default_assets = Path("./assets")
    if not args.assets_dir.exists():
        print(f"Copying {default_assets} to {args.assets_dir}")
        shutil.copytree(default_assets, args.assets_dir)

    bench_dir = args.assets_dir / args.benchmark
    tc_root = bench_dir / "test_cases"
    if not tc_root.exists():
        raise RuntimeError(f"No test_cases/ directory found in {bench_dir}")

    # Step 1: Create query.json
    queries = load_query_definitions(bench_dir)
    query_json_path = bench_dir / "query.json"
    with open(query_json_path, "w") as f:
        json.dump({"seeds": [42], "queries": queries}, f, indent=2)
    print(f"[1/3] Created query.json ({len(queries)} queries)")

    # Step 2: MSA
    query_msa_path = bench_dir / "query_msa.json"
    if not args.skip_msa:
        print("[2/3] Running MSA server...")
        run_msa_batched(bench_dir=bench_dir, queries=queries)
        msa_output_path = bench_dir / "msa" / "query_msa.json"
        if not msa_output_path.exists():
            raise RuntimeError(f"query_msa.json not found in {bench_dir / 'msa'}")
        shutil.move(msa_output_path, query_msa_path)
    else:
        print("[2/3] Skipping MSA (--skip-msa)")

    # Step 3: Dataloader
    batch_dir = bench_dir / "batch"
    print("[3/3] Running dataloader...")
    save_batches(query_msa_path, batch_dir)

    print(f"Done: {bench_dir}")


if __name__ == "__main__":
    main()
