"""USAlign-based structure comparison utilities."""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import biotite.structure as struc
import numpy as np
import torch

from confornet.utils.align import (
    get_backbone_indices_for_ranges,
    get_matching_backbone_indices,
)
from confornet.utils.io import write_cif

if TYPE_CHECKING:
    from confornet.data import PredictionQuery, Reference


def create_backbone_cif_array(
    source_array,
    indices: list[int],
    coords: np.ndarray,
):
    """
    Create a new AtomArray with just the specified backbone atoms and coordinates.

    Residues are renumbered sequentially (1, 2, 3, ...) to enable USAlign's
    index-based matching mode.
    """
    result = source_array[indices].copy()
    result.coord = coords

    # Renumber residues sequentially (1, 2, 3, ...)
    # Each residue has 4 backbone atoms (N, CA, C, O)
    n_residues = len(indices) // 4
    result.res_id = np.repeat(np.arange(1, n_residues + 1), 4)

    return result


def _parse_do_output(output: str) -> list[tuple[int, int, float]]:
    """Parse USAlign -do output to extract per-residue CA-CA distances."""
    distances = []
    in_table = False

    for line in output.split("\n"):
        if "#Aligned atom 1" in line:
            in_table = True
            continue
        if in_table:
            if line.startswith("###"):
                break
            parts = line.split("\t")
            if len(parts) >= 3:
                # Format: " CA  SER A   1."
                mobile_resid = int(parts[0].rstrip(".").split()[-1])
                target_resid = int(parts[1].rstrip(".").split()[-1])
                distance = float(parts[2].strip())
                distances.append((mobile_resid, target_resid, distance))

    return distances


def _run_usalign(
    mobile_cif: Path,
    target_cif: Path,
    usalign_path: str,
    output_distances: bool = False,
) -> tuple[float, list[tuple[int, int, float]] | None]:
    """
    Run USAlign with -TMscore 1 mode (residue index matching).

    Returns:
        (rmsd, distances) where distances is None unless output_distances=True
    """
    cmd = [usalign_path, str(mobile_cif), str(target_cif), "-TMscore", "1", "-ter", "0"]
    if output_distances:
        cmd.append("-do")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    output = result.stdout

    # Parse RMSD
    rmsd_match = re.search(r"RMSD=\s*([\d.]+)", output)
    rmsd = float(rmsd_match.group(1)) if rmsd_match else float("inf")

    distances = None
    if output_distances:
        distances = _parse_do_output(output)

    return rmsd, distances


def _get_metric_indices_in_alignment(
    alignment_ranges: list[tuple[int, int]],
    metric_ranges: list[tuple[int, int]],
) -> set[int]:
    """
    Map metric residues to their sequential positions in USAlign backbone CIF.

    USAlign backbone CIFs are renumbered sequentially (1, 2, 3...) using only
    residues from alignment_ranges. This function finds which sequential CIF
    indices correspond to residues in metric_ranges.
    """
    # Get original sequence positions used for alignment
    align_positions = []
    for start, end in alignment_ranges:
        align_positions.extend(range(start, end))
    align_positions = sorted(set(align_positions))

    # Get original sequence positions for metrics
    metric_positions = set()
    for start, end in metric_ranges:
        metric_positions.update(range(start, end))

    # Map to sequential CIF indices (1-indexed)
    metric_cif_indices = set()
    for cif_idx, orig_pos in enumerate(align_positions, start=1):
        if orig_pos in metric_positions:
            metric_cif_indices.add(cif_idx)

    return metric_cif_indices


def _compute_rmsd_from_distances(
    distances: list[tuple[int, int, float]],
    metric_indices: set[int],
) -> float:
    """Compute RMSD from filtered per-residue distances."""
    metric_dists = [d for m, t, d in distances if m in metric_indices]
    if not metric_dists:
        return float("inf")
    return np.sqrt(np.mean([d**2 for d in metric_dists]))


def _matched_residue_positions(
    align_idx: list[int],
    atom_array,
) -> list[int]:
    """Extract 0-indexed residue sequence positions from matched backbone atom indices."""
    starts = struc.get_residue_starts(atom_array)
    positions = []
    for i in range(0, len(align_idx), 4):
        atom_idx = align_idx[i]
        res_idx = int(np.searchsorted(starts, atom_idx, side="right")) - 1
        positions.append(res_idx)
    return positions


def compute_rmsd_usalign(
    pred_coords: torch.Tensor,
    query: "PredictionQuery",
    reference: "Reference",
    usalign_path: str,
) -> float:
    """Compute RMSD using alignment_resid_ranges for alignment, metric_resid_ranges for RMSD."""
    # Get matched backbone indices for BOTH sides
    align_q_idx, align_r_idx = get_matching_backbone_indices(
        query,
        reference,
        resid_ranges=reference.alignment_resid_ranges,
        query_resid_ranges=query.alignment_resid_ranges,
    )

    # Build both CIFs from matched pairs (guarantees identical residue counts)
    pred_bb_coords = pred_coords[align_q_idx].numpy()
    pred_bb_array = create_backbone_cif_array(
        query.atom_array, align_q_idx, pred_bb_coords
    )

    ref_bb_coords = reference.coordinates[align_r_idx].numpy()
    ref_bb_array = create_backbone_cif_array(
        reference.atom_array, align_r_idx, ref_bb_coords
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pred_bb_cif = Path(tmpdir) / "pred.cif"
        ref_bb_cif = Path(tmpdir) / "ref.cif"
        write_cif(pred_bb_array, pred_bb_array.coord, pred_bb_cif)
        write_cif(ref_bb_array, ref_bb_array.coord, ref_bb_cif)

        if reference.ranges_match:
            rmsd, _ = _run_usalign(pred_bb_cif, ref_bb_cif, usalign_path)
        else:
            _, distances = _run_usalign(
                pred_bb_cif, ref_bb_cif, usalign_path, output_distances=True
            )
            # Compute metric indices from actual matched positions on BOTH sides
            matched_ref_pos = _matched_residue_positions(
                align_r_idx, reference.atom_array
            )
            matched_query_pos = _matched_residue_positions(
                align_q_idx, query.atom_array
            )
            ref_metric = set()
            for s, e in reference.metric_resid_ranges:
                ref_metric.update(range(s, e))
            query_metric = set()
            if query.metric_resid_ranges:
                for s, e in query.metric_resid_ranges:
                    query_metric.update(range(s, e))
            metric_cif_indices = {
                cif_idx
                for cif_idx, (rp, qp) in enumerate(
                    zip(matched_ref_pos, matched_query_pos), start=1
                )
                if rp in ref_metric and (not query_metric or qp in query_metric)
            }
            rmsd = _compute_rmsd_from_distances(distances, metric_cif_indices)

        return rmsd


def prepare_reference_backbone_cifs(
    references: list["Reference"],
    output_dir: Path,
) -> dict[str, Path]:
    """
    Prepare backbone CIFs for all references using alignment_resid_ranges.

    Should be called once before parallelization to avoid redundant I/O.
    """
    ref_cif_paths = {}

    for ref in references:
        # Get backbone indices using ALIGNMENT ranges
        align_r_idx = get_backbone_indices_for_ranges(
            ref.atom_array, ref.alignment_resid_ranges
        )

        # Create reference backbone CIF
        ref_bb_coords = ref.coordinates[align_r_idx].numpy()
        ref_bb_array = create_backbone_cif_array(
            ref.atom_array, align_r_idx, ref_bb_coords
        )

        # Write to output directory
        ref_cif_path = output_dir / f"{ref.name}_backbone.cif"
        write_cif(ref_bb_array, ref_bb_array.coord, ref_cif_path)
        ref_cif_paths[ref.name] = ref_cif_path

    return ref_cif_paths
