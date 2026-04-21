"""Parse sample metadata out of structure-file paths for each evaluation mode."""

import re
from pathlib import Path

from confornet.eval.results import (
    DiversitySampleMetadata,
    SampleMetadata,
    TransferSampleMetadata,
)


def parse_diversity_path(path: Path, base_dir: Path) -> DiversitySampleMetadata | None:
    """
    Extract seed, query_id, step, confornet_idx, sample_idx from diversity path.

    Pattern: {test_case}/seed_{N}/{query_id}_step_{N}_confornet_{N}_sample_{N}.cif
    """
    rel_path = path.relative_to(base_dir)
    parts = rel_path.parts

    seed_match = re.match(r"seed_(\d+)", parts[-2])
    if not seed_match:
        return None
    run_idx = int(seed_match.group(1))

    filename = path.stem
    match = re.match(r"(.+)_step_(\d+)_confornet_(\d+)_sample_(\d+)", filename)
    if not match:
        return None

    return DiversitySampleMetadata(
        path=path,
        sample_idx=int(match.group(4)),
        run_idx=run_idx,
        target_query_id=match.group(1),
        step=int(match.group(2)),
        confornet_idx=int(match.group(3)),
    )


def parse_mse_path(
    path: Path, base_dir: Path, tc_name: str, ref_to_query: dict[str, str]
) -> TransferSampleMetadata | None:
    """
    Extract test_case, ref_name, run_idx, sample_idx from MSE path.

    Pattern: {test_case}/{source_ref_name}/run_{N}/sample_{N}.cif
    """
    rel_path = path.relative_to(base_dir)
    parts = rel_path.parts

    # parts: [test_case, ref_name, run_N, sample_N.cif]
    if len(parts) < 4:
        return None

    source_ref_name = parts[-3]

    # Extract run_idx from directory
    run_match = re.match(r"run_(\d+)", parts[-2])
    if not run_match:
        return None
    run_idx = int(run_match.group(1))

    # Extract sample_idx from filename
    filename = path.stem
    sample_match = re.match(r"sample_(\d+)", filename)
    if not sample_match:
        return None

    query_id = ref_to_query.get(source_ref_name)
    if query_id is None:
        return None

    return TransferSampleMetadata(
        path=path,
        sample_idx=int(sample_match.group(1)),
        run_idx=run_idx,
        target_query_id=query_id,
        source_test_case=tc_name,
        source_ref_name=source_ref_name,
    )


def parse_transfer_path(path: Path, base_dir: Path) -> TransferSampleMetadata | None:
    """
    Extract source_tc, target_tc, target_query_id, ref, run, sample from transfer path.

    Patterns:
      source_{source_tc}/{target_tc}/{target_query_id}/ref_{ref}_run_{N}_sample_{N}.cif
      {target_tc}/{target_query_id}/sample_{N}.cif   (explicit --confornet-path)
    """
    rel_path = path.relative_to(base_dir)
    parts = rel_path.parts

    # Explicit --confornet-path: 3 parts, filename is "sample_{N}.cif"
    if len(parts) == 3:
        m = re.match(r"sample_(\d+)", path.stem)
        if not m:
            return None
        return TransferSampleMetadata(
            path=path,
            sample_idx=int(m.group(1)),
            run_idx=0,
            target_query_id=parts[-2],
            source_test_case="explicit",
            source_ref_name="explicit",
        )

    if len(parts) < 4:
        return None

    # Extract source_tc from parent directory
    source_match = re.match(r"source_(.+)", parts[-4])
    if not source_match:
        return None
    source_tc = source_match.group(1)

    # Extract target_query_id from directory before filename
    target_query_id = parts[-2]

    # Extract ref, run, sample from filename
    filename = path.stem
    match = re.match(r"ref_(.+)_run_(\d+)_sample_(\d+)", filename)
    if not match:
        return None

    return TransferSampleMetadata(
        path=path,
        sample_idx=int(match.group(3)),
        run_idx=int(match.group(2)),
        source_test_case=source_tc,
        source_ref_name=match.group(1),
        target_query_id=target_query_id,
    )


def parse_of3p_path(path: Path, base_dir: Path) -> SampleMetadata | None:
    rel_path = path.relative_to(base_dir)
    parts = rel_path.parts
    if len(parts) < 3:
        return None

    target_query_id = parts[-3]

    filename = path.name
    m = re.match(
        rf"^{re.escape(target_query_id)}(?:_\d+)?_seed_(\d+)_sample_(\d+)_model\.(?:cif|pdb)$",
        filename,
    )

    if not m:
        return None

    return SampleMetadata(
        path=path,
        run_idx=int(m.group(1)),
        sample_idx=int(m.group(2)),
        target_query_id=target_query_id,
    )
