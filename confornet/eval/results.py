"""
Data classes for evaluation results.
"""

import pickle
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from confornet.eval.metrics import benchmark_success_rate


@dataclass
class SampleMetadata:
    """Base metadata for a single sample."""

    path: Path  # Full path to sample CIF
    sample_idx: int  # Sample index within run
    run_idx: int  # Run index (seed for diversity, run_idx for mse/transfer)
    target_query_id: str  # Query ID used for inference


@dataclass
class DiversitySampleMetadata(SampleMetadata):
    """Metadata for diversity mode samples."""

    step: int  # Training step checkpoint
    confornet_idx: int  # Which ConforNet (0 to k-1)


@dataclass
class TransferSampleMetadata(SampleMetadata):
    """Metadata for MSE and transfer mode samples."""

    source_test_case: str  # Test case the ConforNet came from (same as target for MSE)
    source_ref_name: str  # Reference the ConforNet was trained on


@dataclass
class TestCaseResults:
    """All sample results for one test case."""

    test_case: str
    references: list[str]  # Reference names (length R)
    samples: list[SampleMetadata]  # (length N)

    # Metrics as arrays - shape: [n_samples, n_refs]
    rmsd: np.ndarray  # [N, R]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with flattened columns."""
        rows = []
        for i, sample in enumerate(self.samples):
            row = asdict(sample)
            # Convert Path to string for serialization
            row["path"] = str(row["path"])
            for j, ref in enumerate(self.references):
                row[f"rmsd_to_{ref}"] = self.rmsd[i, j]
            rows.append(row)
        return pd.DataFrame(rows)

    def get_min_rmsd(self) -> np.ndarray:
        """Get minimum RMSD to any reference for each sample. Shape: [N]."""
        return np.min(self.rmsd, axis=1)


@dataclass
class EvalResults:
    """Complete evaluation results for a benchmark run."""

    benchmark: str
    mode: str  # 'diversity', 'mse', 'transfer'
    threshold: float  # RMSD threshold used
    test_cases: dict[str, TestCaseResults]

    # Metadata
    samples_dir: str | None = None
    assets_dir: str | None = None

    # Computed on demand
    _success_rates: dict | None = field(default=None, repr=False)

    def save(self, path: Path):
        """Save to pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "EvalResults":
        """Load from pickle."""
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def merge(cls, results_list: list["EvalResults"]) -> "EvalResults":
        """
        Merge multiple EvalResults objects.

        All results must have the same benchmark, mode, and threshold.
        Test cases are concatenated (error on duplicate keys).
        """
        if len(results_list) == 0:
            raise ValueError("Cannot merge empty list")
        if len(results_list) == 1:
            return results_list[0]

        first = results_list[0]
        for r in results_list[1:]:
            if r.benchmark != first.benchmark:
                raise ValueError(
                    f"Benchmark mismatch: {r.benchmark} vs {first.benchmark}"
                )
            if r.mode != first.mode:
                raise ValueError(f"Mode mismatch: {r.mode} vs {first.mode}")
            if r.threshold != first.threshold:
                raise ValueError(
                    f"Threshold mismatch: {r.threshold} vs {first.threshold}"
                )

        # Merge test_cases
        merged_test_cases: dict[str, TestCaseResults] = {}
        for r in results_list:
            for tc_name, tc_results in r.test_cases.items():
                if tc_name in merged_test_cases:
                    raise ValueError(f"Duplicate test case: {tc_name}")
                merged_test_cases[tc_name] = tc_results

        # Collect samples_dir from all results
        samples_dirs = []
        for r in results_list:
            if r.samples_dir:
                if isinstance(r.samples_dir, list):
                    samples_dirs.extend(r.samples_dir)
                else:
                    samples_dirs.append(r.samples_dir)

        return cls(
            benchmark=first.benchmark,
            mode=first.mode,
            threshold=first.threshold,
            test_cases=merged_test_cases,
            samples_dir=samples_dirs
            if len(samples_dirs) > 1
            else (samples_dirs[0] if samples_dirs else None),
            assets_dir=first.assets_dir,
        )

    def filter_by_ref_index(self, ref_idx: int) -> "EvalResults":
        """Return a new EvalResults keeping only one reference column per test case."""
        new_test_cases: dict[str, TestCaseResults] = {}
        for tc_name, tc in self.test_cases.items():
            if ref_idx >= len(tc.references):
                continue
            new_test_cases[tc_name] = TestCaseResults(
                test_case=tc.test_case,
                references=[tc.references[ref_idx]],
                samples=tc.samples,
                rmsd=tc.rmsd[:, ref_idx : ref_idx + 1],
            )
        return EvalResults(
            benchmark=self.benchmark,
            mode=self.mode,
            threshold=self.threshold,
            test_cases=new_test_cases,
            samples_dir=self.samples_dir,
            assets_dir=self.assets_dir,
        )

    def group_by_source(self) -> dict[str, "EvalResults"]:
        """Group transfer-mode results by source test case."""
        source_groups: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        for tc_name, tc_results in self.test_cases.items():
            for i, sample in enumerate(tc_results.samples):
                if isinstance(sample, TransferSampleMetadata):
                    source_groups[sample.source_test_case][tc_name].append((i, sample))

        source_results: dict[str, EvalResults] = {}
        for source_tc, tc_samples in source_groups.items():
            new_test_cases: dict[str, TestCaseResults] = {}
            for tc_name, idx_sample_list in tc_samples.items():
                original_tc = self.test_cases[tc_name]
                indices = [i for i, _ in idx_sample_list]
                samples = [s for _, s in idx_sample_list]
                new_test_cases[tc_name] = TestCaseResults(
                    test_case=tc_name,
                    references=original_tc.references,
                    samples=samples,
                    rmsd=original_tc.rmsd[indices],
                )
            source_results[source_tc] = EvalResults(
                benchmark=self.benchmark,
                mode=self.mode,
                threshold=self.threshold,
                test_cases=new_test_cases,
                samples_dir=self.samples_dir,
                assets_dir=self.assets_dir,
            )
        return source_results

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all test cases to a single DataFrame."""
        dfs = []
        for tc_name, tc_results in self.test_cases.items():
            df = tc_results.to_dataframe()
            df["test_case"] = tc_name
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def compute_success_rates(
        self, k: int = 1, n_bootstrap: int = 100
    ) -> tuple[float, float]:
        """Bootstrap overall success rate @ k. Returns `(mean, std)`."""
        return benchmark_success_rate(self, k=k, n_bootstrap=n_bootstrap)

    def summary(self) -> str:
        """Return a summary string of the results."""
        lines = [
            f"EvalResults for {self.benchmark} ({self.mode} mode)",
            f"  Threshold: {self.threshold} A",
            f"  Test cases: {len(self.test_cases)}",
        ]
        total_samples = sum(len(tc.samples) for tc in self.test_cases.values())
        lines.append(f"  Total samples: {total_samples}")

        for tc_name, tc_results in self.test_cases.items():
            min_rmsds = tc_results.get_min_rmsd()
            lines.append(
                f"  {tc_name}: {len(tc_results.samples)} samples, "
                f"min RMSD: {min_rmsds.min():.2f}, mean: {min_rmsds.mean():.2f}"
            )
        return "\n".join(lines)
