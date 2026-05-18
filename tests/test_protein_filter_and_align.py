"""Tests for filter_to_protein, modified-AA mapping, and local sequence_align."""

from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest

from confornet.utils.align import (
    BACKBONE_ATOMS,
    MODIFIED_AA_3TO1,
    filter_to_protein,
    get_sequence,
    sequence_align,
)
from confornet.utils.io import load_structure

REPO = Path(__file__).parent.parent


def _make_residue(res_id: int, res_name: str, atom_names: list[str]) -> struc.AtomArray:
    n = len(atom_names)
    a = struc.AtomArray(n)
    a.coord = np.zeros((n, 3), dtype=np.float32)
    a.atom_name = np.array(atom_names)
    a.res_id = np.array([res_id] * n)
    a.res_name = np.array([res_name] * n)
    a.chain_id = np.array(["A"] * n)
    a.element = np.array([nm[0] for nm in atom_names])
    return a


def _concat(*arrays: struc.AtomArray) -> struc.AtomArray:
    return sum(arrays[1:], start=arrays[0])


class TestFilterToProtein:
    def test_drops_calcium_ion(self):
        protein = _make_residue(1, "ALA", BACKBONE_ATOMS + ["CB"])
        calcium = _make_residue(2, "CA", ["CA"])  # ion: atom name 'CA', res name 'CA'
        aa = _concat(protein, calcium)
        out = filter_to_protein(aa)
        assert len(out) == 5
        assert set(out.res_name.tolist()) == {"ALA"}

    def test_drops_water(self):
        protein = _make_residue(1, "GLY", BACKBONE_ATOMS)
        water = _make_residue(2, "HOH", ["O"])
        out = filter_to_protein(_concat(protein, water))
        assert set(out.res_name.tolist()) == {"GLY"}

    def test_drops_ligand_without_backbone(self):
        protein = _make_residue(1, "VAL", BACKBONE_ATOMS + ["CB", "CG1"])
        ligand = _make_residue(
            2, "GDP", ["P", "OP1", "OP2", "OP3", "O5'", "C5'", "C4'"]
        )
        out = filter_to_protein(_concat(protein, ligand))
        assert set(out.res_name.tolist()) == {"VAL"}

    def test_keeps_modified_aa_with_full_backbone(self):
        # Phosphoserine has full N, CA, C, O backbone.
        sep = _make_residue(1, "SEP", BACKBONE_ATOMS + ["CB", "OG", "P"])
        out = filter_to_protein(sep)
        assert len(out) == len(sep)

    def test_keeps_all_when_all_protein(self):
        a = _make_residue(1, "ALA", BACKBONE_ATOMS)
        b = _make_residue(2, "GLY", BACKBONE_ATOMS)
        out = filter_to_protein(_concat(a, b))
        assert len(out) == 8

    def test_empty_array_returns_empty(self):
        empty = struc.AtomArray(0)
        out = filter_to_protein(empty)
        assert len(out) == 0

    def test_interspersed_waters_dropped(self):
        # Waters between protein residues — residue indices in get_residue_starts
        # would shift if we don't filter. Verify filter removes them.
        a = _make_residue(1, "ALA", BACKBONE_ATOMS)
        w = _make_residue(2, "HOH", ["O"])
        b = _make_residue(3, "GLY", BACKBONE_ATOMS)
        out = filter_to_protein(_concat(a, w, b))
        assert set(out.res_name.tolist()) == {"ALA", "GLY"}
        assert len(out) == 8

    def test_2nxq_b_calcium_filtered(self):
        # Real foldswitching reference: 2NXQ_B has 2 Ca²⁺ ions + ACT + HOH residues.
        path = REPO / "assets/foldswitching/test_cases/fs-1jfk_A-2nxq_B/reference/2NXQ_B.pdb"
        aa = load_structure(path)
        out = filter_to_protein(aa)
        seq = get_sequence(out)
        assert len(seq) == 64
        assert "X" not in seq
        assert seq.endswith("FYGSI")


class TestGetSequenceModifiedAA:
    @pytest.mark.parametrize(
        "res_name,expected_letter",
        [
            ("SEP", "S"),
            ("TPO", "T"),
            ("PTR", "Y"),
            ("MSE", "M"),
            ("OCS", "C"),
            ("CSD", "C"),
            ("YCM", "C"),
        ],
    )
    def test_modified_aa_maps_to_parent(self, res_name, expected_letter):
        a = _make_residue(1, res_name, BACKBONE_ATOMS)
        assert get_sequence(a) == expected_letter

    def test_standard_aa_unchanged(self):
        seq_codes = [("ALA", "A"), ("GLY", "G"), ("VAL", "V"), ("LYS", "K")]
        for three, one in seq_codes:
            a = _make_residue(1, three, BACKBONE_ATOMS)
            assert get_sequence(a) == one

    def test_unknown_residue_falls_back_to_X(self):
        a = _make_residue(1, "XYZ", BACKBONE_ATOMS)
        assert get_sequence(a) == "X"

    def test_modified_map_keys_are_uppercase(self):
        # Sanity: MODIFIED_AA_3TO1 keys must be uppercase or we miss lookups.
        for k in MODIFIED_AA_3TO1:
            assert k == k.upper()


class TestSequenceAlignLocal:
    def test_short_ref_against_long_sample_one_block(self):
        # Long sample with the ref-sequence embedded in the middle, then garbage
        # trailing. Global alignment scatters the trailing ref residues across
        # the garbage; local alignment must return ONE contiguous block.
        ref = "AEALFKEIDVNGDGAVSYEEVKAF"
        sample = "MGSSHHHHHH" + ref + "QGQDLSDDKIGLKVLLKAFKKGADIFFSALDADGSG"
        matched = sequence_align(sample, ref)
        # Sample indices should be contiguous (one run of length len(ref))
        sample_idxs = [s for s, _ in matched]
        assert sample_idxs == list(range(sample_idxs[0], sample_idxs[0] + len(ref)))
        # Ref indices should be contiguous starting at 0
        ref_idxs = [r for _, r in matched]
        assert ref_idxs == list(range(0, len(ref)))

    def test_long_sample_short_ref_no_phantom_matches(self):
        # The previously-broken case: 134-res sample with N-terminal his-tag
        # vs 64-res ref. Global alignment scattered ~30 residues; local must
        # produce a single contiguous block.
        ref = "AEALFKEIDVNGDGAVSYEEVKAFVSKKRAIKNEQLLQLIFKSIDADGNGEIDQNEFAKFYGSI"
        sample = (
            "MAEALFKEIDVNGDGAVSYEEVKAFVSKKRAIKNEQLLQLIFKSIDADGNGEIDQNEFAKFYGSI"
            "QGQDLSDDKIGLKVLLKAFKKGADIFFSALDADGSGKLEKAEFAALYRSFFGEPDRGEEHRRLAEVFRTLDE"
        )
        matched = sequence_align(sample, ref)
        sample_idxs = [s for s, _ in matched]
        # All matches in one contiguous block of length len(ref)
        assert len(matched) == len(ref)
        assert sample_idxs == list(range(sample_idxs[0], sample_idxs[0] + len(ref)))

    def test_equal_length_identical_sequences(self):
        seq = "MKTAYIAKQRQISFVKSHFSRQLEERLG"
        matched = sequence_align(seq, seq)
        assert matched == [(i, i) for i in range(len(seq))]

    def test_sample_with_n_terminal_histag_aligns_to_body(self):
        # Confornet sample format: GS-tag + protein. Local should skip the tag.
        body = "AISGSLIERINNKGTVTVGTEGTYAPFTYHDKDGKLTGYDVE"
        sample = "MGHHHHHHHHHHSSGHIDDDDKHM" + body
        ref = body
        matched = sequence_align(sample, ref)
        assert len(matched) == len(body)
        # First match should land past the his-tag, not inside it.
        first_sample_idx = matched[0][0]
        assert first_sample_idx >= len("MGHHHHHHHHHHSSGHIDDDDKHM")


class TestEndToEndReferenceLoading:
    """Real reference files: exact post-filter lengths and no X residues."""

    @pytest.mark.parametrize(
        "ref_path,unfiltered_len,filtered_len",
        [
            # 2NXQ_B: 64 protein + 2 Ca²⁺ (atom-named 'CA') — filter drops the ions.
            ("assets/foldswitching/test_cases/fs-1jfk_A-2nxq_B/reference/2NXQ_B.pdb", 66, 64),
            # HCK/3VS1_A: 425 protein (incl. PTR) + 1 Ca²⁺ — filter drops the ion;
            # PTR survives (full backbone) and maps to Y via MODIFIED_AA_3TO1.
            ("assets/kinases/test_cases/HCK/reference/3VS1_A.cif", 426, 425),
            # AKT1/6BUU_B: 319 residues incl. SEP+TPO (modified AAs, kept).
            ("assets/kinases/test_cases/AKT1/reference/6BUU_B.cif", 319, 319),
            # MTR1A/6ME2_A: 267 residues incl. YCM (modified Cys, kept).
            ("assets/gpcr/test_cases/MTR1A/reference/6ME2_A.cif", 267, 267),
        ],
    )
    def test_reference_clean_after_filter(self, ref_path, unfiltered_len, filtered_len):
        aa = load_structure(REPO / ref_path)
        assert len(get_sequence(aa)) == unfiltered_len

        seq = get_sequence(filter_to_protein(aa))
        assert len(seq) == filtered_len
        assert "X" not in seq
