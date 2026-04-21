"""MSA + OF3p batch construction — shared by preprocess.py (benchmarks) and scripts/diverse.py, scripts/supervised.py (single sequences)."""

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import torch
from openfold3.core.data.framework.data_module import DataModule, DataModuleConfig
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessorSettings,
)
from openfold3.projects.of3_all_atom.config.dataset_config_components import (
    MSASettings,
    TemplateSettings,
)
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    InferenceDatasetSpec,
    InferenceJobConfig,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)
from tqdm import tqdm

from confornet.data import PredictionQuery
from confornet.utils.dist import get_local_rank

MSA_FILEPATH_KEYS = (
    "main_msa_file_paths",
    "paired_msa_file_paths",
    "template_alignment_file_path",
)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZUO")


def run_msa_batched(
    bench_dir: Path,
    queries: dict,
    batch_size: int = 30,
    seeds: list[int] = [42],
) -> Path:
    bench_dir = Path(bench_dir).resolve()
    assets_dir = bench_dir.parent.resolve()
    batches_root, final_dir = bench_dir / "msa_batches", bench_dir / "msa"
    subs = ("main", "mappings", "paired", "raw", "template")
    for d in (batches_root, final_dir, *[final_dir / s for s in subs]):
        d.mkdir(parents=True, exist_ok=True)

    def move_top(src: Path, dst: Path):
        if src.exists():
            for x in src.iterdir():
                y = dst / x.name
                if not y.exists():
                    shutil.move(str(x), str(y))

    def rewrite_paths(q: dict, outdir: Path):
        def abs_in_final(p: str) -> str:
            if not p:
                return p
            p = str(p).lstrip("./")
            k = p.rfind("assets/")
            if k != -1:
                p = p[k:]
            P = Path(p)
            P = P.resolve() if P.is_absolute() else (outdir / p).resolve()

            s = str(P)
            tail = (
                s.split("/msa/")[-1].lstrip("/")
                if "/msa/" in s
                else str(P.relative_to(outdir))
            )
            return str((final_dir / tail).resolve())

        for c in q.get("chains", []):
            for key in MSA_FILEPATH_KEYS:
                v = c.get(key)
                if not v:
                    continue
                c[key] = (
                    [abs_in_final(x) for x in v]
                    if isinstance(v, list)
                    else abs_in_final(v)
                )

    items = list(queries.items())
    merged: dict = {}

    for i in range(0, len(items), batch_size):
        bi = i // batch_size
        bdir, outdir = (
            batches_root / f"batch_{bi:04d}",
            (batches_root / f"batch_{bi:04d}" / "msa"),
        )
        outdir.mkdir(parents=True, exist_ok=True)

        qpath = bdir / "query.json"
        qpath.write_text(
            json.dumps(
                {"seeds": seeds, "queries": dict(items[i : i + batch_size])}, indent=2
            )
        )

        cmd = [
            "python",
            "-m",
            "openfold3.run_openfold",
            "align-msa-server",
            "--query_json",
            str(qpath),
            "--output_dir",
            str(outdir),
        ]
        subprocess.run(cmd, check=True, cwd=str(assets_dir))

        mpath = outdir / "query_msa.json"
        if not mpath.exists():
            raise RuntimeError(f"Batch {bi:04d}: missing {mpath}")
        j = json.loads(mpath.read_text())

        for name, q in j.get("queries", {}).items():
            if name in merged:
                raise RuntimeError(f"Duplicate query key while merging: {name}")
            rewrite_paths(q, outdir)
            merged[name] = q

        for s in subs:
            move_top(outdir / s, final_dir / s)

    out = final_dir / "query_msa.json"
    out.write_text(json.dumps({"seeds": seeds, "queries": merged}, indent=2))
    shutil.rmtree(batches_root, ignore_errors=True)
    return out


def save_batches(query_json: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    query_set = InferenceQuerySet.from_json(query_json)
    cfg = InferenceJobConfig(
        query_set=query_set,
        seeds=[42],
        msa=MSASettings(),
        template=TemplateSettings(),
        template_preprocessor_settings=TemplatePreprocessorSettings(mode="predict"),
    )
    dm = DataModule(
        DataModuleConfig(
            datasets=[InferenceDatasetSpec(config=cfg)],
            batch_size=1,
            num_workers=0,
        ),
    )
    dm.setup()
    for batch in tqdm(dm.predict_dataloader(), desc="batch"):
        torch.save(batch, output_dir / f"{batch['query_id'][0]}.pt")


def load_query_definitions(bench_dir: Path) -> dict[str, dict]:
    queries = {}
    for tc_dir in sorted((bench_dir / "test_cases").iterdir()):
        query_dir = tc_dir / "query"
        if query_dir.exists():
            for f in sorted(query_dir.glob("*.json")):
                queries[f.stem] = json.loads(f.read_text())
    return queries


def read_sequence(sequence: str) -> tuple[str, str]:
    """Validate a raw AA string and return (default_name, sequence)."""
    seq = sequence.strip().upper()
    if not seq or not set(seq).issubset(VALID_AA):
        raise ValueError(f"--sequence is not a valid amino-acid string: {sequence!r}")
    return f"seq_{hashlib.sha1(seq.encode()).hexdigest()[:8]}", seq


def prepare_sequence_batch(sequence: str, cache_dir: Path, name: str) -> Path:
    """Build (and cache) an OF3p batch for a single protein sequence.

    Layout: {cache_dir}/{name}/batch/{name}.pt (plus MSA artifacts alongside).
    Rank-0 does the work; other ranks wait on a .done sentinel.
    """
    bench_dir = Path(cache_dir).resolve() / name
    batch_path = bench_dir / "batch" / f"{name}.pt"
    done = bench_dir / ".done"

    if done.exists() and batch_path.exists():
        return batch_path

    if get_local_rank() != 0:
        while not (done.exists() and batch_path.exists()):
            time.sleep(2)
        return batch_path

    bench_dir.mkdir(parents=True, exist_ok=True)
    queries = {
        name: {
            "chains": [
                {
                    "molecule_type": "PROTEIN",
                    "chain_ids": ["A"],
                    "sequence": sequence,
                }
            ]
        }
    }
    query_json = bench_dir / "query.json"
    query_json.write_text(json.dumps({"seeds": [42], "queries": queries}, indent=2))

    run_msa_batched(bench_dir=bench_dir, queries=queries)
    query_msa = bench_dir / "query_msa.json"
    shutil.move(bench_dir / "msa" / "query_msa.json", query_msa)

    save_batches(query_msa, bench_dir / "batch")
    if not batch_path.exists():
        raise RuntimeError(f"Batch not produced at {batch_path}")

    done.touch()
    return batch_path


def load_sequence_query(
    sequence: str, cache_dir: Path, name: str, device: torch.device
) -> PredictionQuery:
    return PredictionQuery.from_batch_file(
        prepare_sequence_batch(sequence, cache_dir, name), device
    )
