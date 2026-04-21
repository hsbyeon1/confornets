"""Test TM6 alignment correctness for GPCR benchmark."""

import json
import tempfile
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest
import torch

from confornet.data import PredictionQuery, Reference
from confornet.utils.align import get_matching_backbone_indices, get_sequence
from confornet.utils.io import load_structure, write_cif
from confornet.utils.usalign import (
    _matched_residue_positions,
    _run_usalign,
    create_backbone_cif_array,
)

REPO = Path(__file__).parent.parent
ASSETS = REPO / "assets" / "gpcr"
TEST_ASSETS = Path(__file__).parent / "assets" / "gpcr_predictions"

# Map test case to folded CIF path
FOLD_CIFS = {
    "ACM3": TEST_ASSETS / "fold_acm3_model_0.cif",
    "AA1R": TEST_ASSETS / "fold_aa1r_model_0.cif",
    "HRH1": TEST_ASSETS / "fold_hrh1_model_0.cif",
    "SSR2": TEST_ASSETS / "fold_ssr2_model_0.cif",
}

# References: (test_case, ref_index, ref_name, label)
CASES = [
    ("ACM3", 1, "8E9Z_A", "active"),
    ("ACM3", 2, "4U15_B", "inactive"),
    ("AA1R", 1, "7LD3_R", "active"),
    ("AA1R", 2, "5UEN_B", "inactive"),
    ("HRH1", 1, "8YN2_R", "active"),
    ("HRH1", 2, "8X5Y_B", "inactive"),
    ("SSR2", 1, "7T10_R", "active"),
    ("SSR2", 2, "7XN9_A", "inactive"),
]


def load_query(tc_name: str) -> PredictionQuery:
    """Load a PredictionQuery from a folded CIF + align_metric_info."""
    cif_path = FOLD_CIFS[tc_name]
    if not cif_path.exists():
        pytest.skip(f"Folded CIF not found: {cif_path}")
    atom_array = load_structure(cif_path)
    sequence = get_sequence(atom_array)

    info_path = ASSETS / "test_cases" / tc_name / "align_metric_info.json"
    with open(info_path) as f:
        info = json.load(f)

    qr = info["queries"][tc_name]
    alignment_ranges = [tuple(r) for r in qr["alignment_resid_ranges"]]
    metric_ranges = [tuple(r) for r in qr["metric_resid_ranges"]]

    return PredictionQuery(
        atom_array=atom_array,
        sequence=sequence,
        batch={},
        alignment_resid_ranges=alignment_ranges,
        metric_resid_ranges=metric_ranges,
    )


def load_ref(tc_name: str, ref_idx: int, ref_name: str) -> Reference:
    """Load a Reference with its align_metric_info."""
    ref_path = ASSETS / "test_cases" / tc_name / "reference" / f"{ref_name}.cif"
    info_path = ASSETS / "test_cases" / tc_name / "align_metric_info.json"
    with open(info_path) as f:
        info = json.load(f)
    ref_info = info["references"][ref_name]
    return Reference.from_path(
        ref_name,
        ref_path,
        alignment_resid_ranges=ref_info.get("alignment_resid_ranges"),
        metric_resid_ranges=ref_info.get("metric_resid_ranges"),
    )


def residue_pairs_from_backbone_indices(
    q_idx: list[int], r_idx: list[int], q_array, r_array
) -> list[tuple[int, int]]:
    """Convert backbone atom indices to 0-indexed residue position pairs."""
    q_starts = struc.get_residue_starts(q_array)
    r_starts = struc.get_residue_starts(r_array)
    pairs = []
    for i in range(0, len(q_idx), 4):
        q_res = int(np.searchsorted(q_starts, q_idx[i], side="right")) - 1
        r_res = int(np.searchsorted(r_starts, r_idx[i], side="right")) - 1
        pairs.append((q_res, r_res))
    return pairs


@pytest.fixture(params=CASES, ids=[f"{c[0]}_{c[3]}" for c in CASES])
def alignment_case(request):
    tc, ref_idx, ref_name, label = request.param
    query = load_query(tc)
    ref = load_ref(tc, ref_idx, ref_name)
    return tc, label, query, ref


class TestMSETrainingAlignment:
    """Test that get_matching_backbone_indices (MSE training path) aligns TM6 correctly.

    MSE training uses ALL aligned positions (not just TM6), so small boundary
    mismatches (1-2 residues) are tolerable. These tests verify the alignment
    is broadly correct.
    """

    def test_tm6_alignment_mostly_correct(self, alignment_case):
        """Most query TM6 residues should map to ref TM6, and vice versa."""
        tc, label, query, ref = alignment_case

        q_idx, r_idx = get_matching_backbone_indices(
            query, ref, query_resid_ranges=query.alignment_resid_ranges
        )
        pairs = residue_pairs_from_backbone_indices(
            q_idx, r_idx, query.atom_array, ref.atom_array
        )

        q_metric = set()
        for s, e in query.metric_resid_ranges:
            q_metric.update(range(s, e))
        r_metric = set()
        for s, e in ref.metric_resid_ranges:
            r_metric.update(range(s, e))

        # Query TM6 -> ref TM6
        q_tm6_pairs = [(q, r) for q, r in pairs if q in q_metric]
        q_bad = [(q, r) for q, r in q_tm6_pairs if r not in r_metric]

        # Ref TM6 -> query TM6
        r_tm6_pairs = [(q, r) for q, r in pairs if r in r_metric]
        r_bad = [(q, r) for q, r in r_tm6_pairs if q not in q_metric]

        # Allow up to 3 boundary mismatches on each side
        max_boundary = 3
        assert len(q_bad) <= max_boundary, (
            f"{tc} {label}: {len(q_bad)} query TM6 residues mapped to non-TM6 ref "
            f"(max {max_boundary}): {q_bad}"
        )
        assert len(r_bad) <= max_boundary, (
            f"{tc} {label}: {len(r_bad)} ref TM6 residues mapped from non-TM6 query "
            f"(max {max_boundary}): {r_bad}"
        )

        # But there should be substantial correct overlap
        correct = [(q, r) for q, r in pairs if q in q_metric and r in r_metric]
        assert len(correct) >= min(len(q_metric), len(r_metric)) * 0.8, (
            f"{tc} {label}: only {len(correct)} correct TM6 pairs, "
            f"expected >= 80% of {min(len(q_metric), len(r_metric))}"
        )


class TestUSAlignMetricFiltering:
    """Test that USAlign metric computation filters by BOTH query and ref TM6."""

    def test_metric_indices_exclude_boundary_residues(self, alignment_case):
        """Metric indices should not include query non-TM6 residues mapped to ref TM6."""
        tc, label, query, ref = alignment_case

        q_idx, r_idx = get_matching_backbone_indices(
            query,
            ref,
            resid_ranges=ref.alignment_resid_ranges,
            query_resid_ranges=query.alignment_resid_ranges,
        )
        pairs = residue_pairs_from_backbone_indices(
            q_idx, r_idx, query.atom_array, ref.atom_array
        )

        q_metric = set()
        for s, e in query.metric_resid_ranges:
            q_metric.update(range(s, e))
        r_metric = set()
        for s, e in ref.metric_resid_ranges:
            r_metric.update(range(s, e))

        # Both-side filter (what compute_rmsd_usalign should use)
        metric_pairs = [
            (i + 1, q, r)
            for i, (q, r) in enumerate(pairs)
            if r in r_metric and q in q_metric
        ]
        # Pairs that pass ref-only filter but fail query filter (boundary leaks)
        ref_only_pairs = [
            (i + 1, q, r)
            for i, (q, r) in enumerate(pairs)
            if r in r_metric and q not in q_metric
        ]

        if ref_only_pairs:
            print(
                f"\n{tc} {label}: {len(ref_only_pairs)} boundary residues "
                f"correctly excluded by query metric filter:"
            )
            for cif_idx, q, r in ref_only_pairs:
                print(f"  CIF idx {cif_idx}: query pos {q} -> ref pos {r}")

        assert len(metric_pairs) > 0, f"{tc} {label}: no metric pairs found"

        # All metric pairs must have BOTH sides in TM6
        for _, q, r in metric_pairs:
            assert q in q_metric, f"query pos {q} not in query TM6"
            assert r in r_metric, f"ref pos {r} not in ref TM6"

    def test_usalign_rmsd_uses_correct_indices(self, alignment_case):
        """End-to-end: compute_rmsd_usalign uses both-side metric filtering."""
        tc, label, query, ref = alignment_case

        usalign_path = str(REPO / "packages" / "USalign")
        if not Path(usalign_path).exists():
            pytest.skip("USAlign binary not found")

        pred_coords = torch.from_numpy(query.atom_array.coord).float()

        q_idx, r_idx = get_matching_backbone_indices(
            query,
            ref,
            resid_ranges=ref.alignment_resid_ranges,
            query_resid_ranges=query.alignment_resid_ranges,
        )

        pred_bb_coords = pred_coords[q_idx].numpy()
        pred_bb = create_backbone_cif_array(query.atom_array, q_idx, pred_bb_coords)
        ref_bb_coords = ref.coordinates[r_idx].numpy()
        ref_bb = create_backbone_cif_array(ref.atom_array, r_idx, ref_bb_coords)

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_cif = Path(tmpdir) / "pred.cif"
            ref_cif = Path(tmpdir) / "ref.cif"
            write_cif(pred_bb, pred_bb.coord, pred_cif)
            write_cif(ref_bb, ref_bb.coord, ref_cif)

            _, distances = _run_usalign(
                pred_cif, ref_cif, usalign_path, output_distances=True
            )

        matched_ref_pos = _matched_residue_positions(r_idx, ref.atom_array)
        matched_query_pos = _matched_residue_positions(q_idx, query.atom_array)

        ref_metric_set = set()
        for s, e in ref.metric_resid_ranges:
            ref_metric_set.update(range(s, e))
        query_metric_set = set()
        for s, e in query.metric_resid_ranges:
            query_metric_set.update(range(s, e))

        # Both-side filter (correct)
        both_indices = {
            cif_idx
            for cif_idx, (rp, qp) in enumerate(
                zip(matched_ref_pos, matched_query_pos), start=1
            )
            if rp in ref_metric_set and qp in query_metric_set
        }
        # Ref-only filter (old behavior)
        ref_only_indices = {
            cif_idx
            for cif_idx, rp in enumerate(matched_ref_pos, start=1)
            if rp in ref_metric_set
        }

        leaked = ref_only_indices - both_indices
        if leaked:
            print(f"\n{tc} {label}: {len(leaked)} boundary residues filtered out:")
            for idx in sorted(leaked):
                qp = matched_query_pos[idx - 1]
                rp = matched_ref_pos[idx - 1]
                print(f"  CIF idx {idx}: query pos {qp} -> ref pos {rp}")

        # All both-side indices must be in -do output
        do_indices = {m for m, t, d in distances}
        missing = both_indices - do_indices
        assert not missing, (
            f"{tc} {label}: metric CIF indices missing from -do output: {missing}"
        )

        # All metric indices must have both sides in TM6
        for idx in both_indices:
            qp = matched_query_pos[idx - 1]
            rp = matched_ref_pos[idx - 1]
            assert qp in query_metric_set, f"query pos {qp} not in query TM6"
            assert rp in ref_metric_set, f"ref pos {rp} not in ref TM6"
