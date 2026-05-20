import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import biotite.structure as struc
import pandas as pd
import torch
import yaml

from confornet.utils.align import filter_to_protein, get_sequence
from confornet.utils.io import load_structure

logger = logging.getLogger(__name__)


def _parse_ranges(ranges: list | None) -> list[tuple[int, int]] | None:
    return [tuple(r) for r in ranges] if ranges else None


def _ref_path(ref_dir: Path, name: str) -> Path:
    pdb = ref_dir / f"{name}.pdb"
    return pdb if pdb.exists() else ref_dir / f"{name}.cif"


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        return {"defaults": {"rmsd_threshold": 3.0}, "benchmarks": {}}
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_benchmark_threshold(config: dict, benchmark_name: str) -> float:
    """Get RMSD threshold for a specific benchmark."""
    benchmarks = config.get("benchmarks", {})
    if benchmark_name in benchmarks:
        return benchmarks[benchmark_name].get(
            "rmsd_threshold", config["defaults"]["rmsd_threshold"]
        )
    return config["defaults"]["rmsd_threshold"]


@dataclass
class Reference:
    name: str
    path: Path
    atom_array: struc.AtomArray
    sequence: str
    coordinates: torch.Tensor
    alignment_resid_ranges: list[tuple[int, int]] = None
    metric_resid_ranges: list[tuple[int, int]] = None
    ranges_match: bool = True

    @classmethod
    def from_path(
        cls,
        name: str,
        path: Path,
        alignment_resid_ranges: list | None = None,
        metric_resid_ranges: list | None = None,
    ):
        atom_array = load_structure(path)
        atom_array = filter_to_protein(atom_array)
        sequence = get_sequence(atom_array)
        coordinates = torch.from_numpy(atom_array.coord).float()

        alignment_ranges = (
            [tuple(r) for r in alignment_resid_ranges]
            if alignment_resid_ranges
            else [(0, len(sequence))]
        )
        metric_ranges = (
            [tuple(r) for r in metric_resid_ranges]
            if metric_resid_ranges
            else [(0, len(sequence))]
        )
        ranges_match = alignment_ranges == metric_ranges

        return cls(
            name,
            path,
            atom_array,
            sequence,
            coordinates,
            alignment_ranges,
            metric_ranges,
            ranges_match,
        )


@dataclass
class TestCase:
    name: str
    references: list[Reference]
    query_ids: list[str]
    query_ranges: dict[str, dict] = field(default_factory=dict)


@dataclass
class Benchmark:
    name: str
    assets_dir: Path
    test_cases: list[TestCase]
    config: dict = field(default_factory=dict)

    @classmethod
    def load(cls, name: str, assets_dir: Path):
        assets = assets_dir / name
        refs_csv = pd.read_csv(assets / "references.csv")

        config_path = assets_dir / "config.yaml"
        config = load_config(config_path)

        ref_cols = [c for c in refs_csv.columns if c.startswith("pdbidchain")]

        test_cases = []
        query_to_tc = {}
        for tc_name in refs_csv["test_case"].unique():
            row = refs_csv[refs_csv["test_case"] == tc_name].iloc[0]

            tc_dir = assets / "test_cases" / tc_name
            ref_dir = tc_dir / "reference"

            # Load align_metric_info.json if it exists
            info_path = tc_dir / "align_metric_info.json"
            info = {"queries": {}, "references": {}}
            if info_path.exists():
                with open(info_path) as f:
                    info = json.load(f)

            # Load references. Missing PDB/CIF files are silently skipped — this
            # enables inference-only runs (diversity / transfer / baseline) that
            # never read reference coordinates. Scripts that need refs
            # (run_mse_training, evaluate) will error on empty reference lists.
            references = []
            for col in ref_cols:
                ref_name = row[col]
                if not pd.notna(ref_name):
                    continue
                ref_path = _ref_path(ref_dir, ref_name)
                if not ref_path.exists():
                    continue
                ref_info = info.get("references", {}).get(ref_name, {})
                references.append(
                    Reference.from_path(
                        ref_name,
                        ref_path,
                        alignment_resid_ranges=ref_info.get("alignment_resid_ranges"),
                        metric_resid_ranges=ref_info.get("metric_resid_ranges"),
                    )
                )

            # Discover query_ids from test_cases/{tc}/query/*.json
            query_dir = tc_dir / "query"
            query_ids = sorted(f.stem for f in query_dir.glob("*.json"))
            if not query_ids:
                query_ids = [tc_name]

            # Multi-query → each query must correspond to a named reference. Only
            # enforced when references are present (inference-only assets allowed).
            if len(query_ids) > 1 and references:
                ref_names = {r.name for r in references}
                if set(query_ids) != ref_names:
                    raise ValueError(
                        f"{tc_name}: multi-query test cases must name each query "
                        f"after its reference PDB. Got query_ids={query_ids}, "
                        f"references={sorted(ref_names)}"
                    )

            for qid in query_ids:
                query_to_tc[qid] = tc_name

            query_ranges = info.get("queries", {})

            test_cases.append(TestCase(tc_name, references, query_ids, query_ranges))

        obj = cls(name, assets_dir, test_cases, config)
        obj._query_to_tc = query_to_tc
        return obj

    @property
    def threshold(self) -> float:
        """Get RMSD threshold for this benchmark."""
        return get_benchmark_threshold(self.config, self.name)

    def get_test_case(self, name: str) -> TestCase:
        for tc in self.test_cases:
            if tc.name == name:
                return tc
        raise KeyError(name)

    def get_batch_path(self, query_id: str) -> Path:
        return self.assets_dir / self.name / "batch" / f"{query_id}.pt"

    def get_query_json_path(self, query_id: str) -> Path:
        tc_name = self._query_to_tc[query_id]
        return (
            self.assets_dir
            / self.name
            / "test_cases"
            / tc_name
            / "query"
            / f"{query_id}.json"
        )


@dataclass
class PredictionQuery:
    atom_array: struc.AtomArray
    sequence: str
    batch: dict
    alignment_resid_ranges: list[tuple[int, int]] = None
    metric_resid_ranges: list[tuple[int, int]] = None

    @classmethod
    def from_batch_file(
        cls,
        path: Path,
        device: torch.device | None = None,
        alignment_resid_ranges: list | None = None,
        metric_resid_ranges: list | None = None,
    ):
        batch = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_batch_dict(
            batch, device, alignment_resid_ranges, metric_resid_ranges
        )

    @classmethod
    def from_batch_dict(
        cls,
        batch: dict,
        device: torch.device | None = None,
        alignment_resid_ranges: list | None = None,
        metric_resid_ranges: list | None = None,
    ):
        atom_array = batch.pop("atom_array", None)
        if isinstance(atom_array, list):
            atom_array = atom_array[0]
        batch.pop("query_id", None)
        sequence = get_sequence(atom_array)

        if device:
            batch = _move_to_device(batch, device)

        return cls(
            atom_array,
            sequence,
            batch,
            _parse_ranges(alignment_resid_ranges),
            _parse_ranges(metric_resid_ranges),
        )


def _move_to_device(batch: dict, device: torch.device) -> dict:
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, dict):
            result[k] = {
                kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        else:
            result[k] = v
    return result


def load_test_cases(
    benchmark: Benchmark,
    test_case_str: str | None,
    num_nodes: int = 1,
    node_idx: int = 0,
):
    if test_case_str:
        tc_names = [tc.strip() for tc in test_case_str.split(",")]
        test_cases = [benchmark.get_test_case(tc) for tc in tc_names]
    else:
        test_cases = benchmark.test_cases

    if num_nodes > 1:
        all_tc = test_cases
        test_cases = [tc for i, tc in enumerate(all_tc) if i % num_nodes == node_idx]
        logger.info(
            f"Node {node_idx}/{num_nodes}: {len(test_cases)}/{len(all_tc)} test cases"
        )
    return test_cases
