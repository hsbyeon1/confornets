"""Tests for USAlign-based RMSD computation helpers."""

import numpy as np
from unittest.mock import MagicMock

import biotite.structure as struc

from confornet.utils.align import BACKBONE_ATOMS, get_matching_backbone_indices
from confornet.utils.usalign import create_backbone_cif_array


def create_mock_atom_array(n_residues: int) -> struc.AtomArray:
    """Create a mock AtomArray with specified number of residues."""
    atoms_per_residue = 4  # N, CA, C, O
    n_atoms = n_residues * atoms_per_residue

    atom_array = struc.AtomArray(n_atoms)
    atom_array.coord = np.random.randn(n_atoms, 3)

    # Set backbone atom names and residue IDs
    atom_names = []
    res_ids = []
    res_names = []
    for i in range(n_residues):
        for atom_name in BACKBONE_ATOMS:
            atom_names.append(atom_name)
            res_ids.append(i + 10)  # Start from 10 to test renumbering
            res_names.append("ALA")

    atom_array.atom_name = np.array(atom_names)
    atom_array.res_id = np.array(res_ids)
    atom_array.res_name = np.array(res_names)
    atom_array.chain_id = np.array(["A"] * n_atoms)
    atom_array.element = np.array(["C"] * n_atoms)  # Simplified

    return atom_array


class TestCreateBackboneCifArray:
    """Tests for create_backbone_cif_array function."""

    def test_correct_indices_selected(self):
        """Verify only specified indices are included in output."""
        atom_array = create_mock_atom_array(10)
        original_coords = atom_array.coord.copy()

        # Select only residues 2-4 (indices 8-19, since 4 atoms per residue)
        indices = list(range(8, 20))
        new_coords = original_coords[indices]

        result = create_backbone_cif_array(atom_array, indices, new_coords)

        assert len(result) == 12  # 3 residues * 4 atoms
        np.testing.assert_array_equal(result.coord, new_coords)

    def test_residues_renumbered_sequentially(self):
        """Verify residues become 1, 2, 3, ..."""
        atom_array = create_mock_atom_array(10)

        # Select residues 5-7 (original res_ids: 15, 16, 17)
        indices = list(range(20, 32))  # 3 residues
        new_coords = atom_array.coord[indices]

        result = create_backbone_cif_array(atom_array, indices, new_coords)

        # Should be renumbered to 1, 2, 3
        expected_res_ids = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        np.testing.assert_array_equal(result.res_id, expected_res_ids)

    def test_backbone_atom_order_preserved(self):
        """Verify N, CA, C, O order per residue."""
        atom_array = create_mock_atom_array(5)

        # Select all atoms
        indices = list(range(20))
        new_coords = atom_array.coord[indices]

        result = create_backbone_cif_array(atom_array, indices, new_coords)

        # Check that each residue has N, CA, C, O in order
        for i in range(5):
            start = i * 4
            expected_atoms = ["N", "CA", "C", "O"]
            actual_atoms = list(result.atom_name[start : start + 4])
            assert actual_atoms == expected_atoms, f"Residue {i}: {actual_atoms}"

    def test_coordinates_replaced(self):
        """Verify coordinates are replaced with new values."""
        atom_array = create_mock_atom_array(3)

        indices = list(range(12))
        new_coords = np.ones((12, 3)) * 42.0

        result = create_backbone_cif_array(atom_array, indices, new_coords)

        np.testing.assert_array_equal(result.coord, new_coords)


class TestGetMatchingBackboneIndices:
    """Tests for get_matching_backbone_indices with resid_ranges."""

    def test_metric_ranges_filter_correctly(self):
        """Verify get_matching_backbone_indices filters by resid_ranges."""
        # Create mock query and reference
        query = MagicMock()
        query.atom_array = create_mock_atom_array(10)
        query.sequence = "AAAAAAAAAA"

        reference = MagicMock()
        reference.atom_array = create_mock_atom_array(10)
        reference.sequence = "AAAAAAAAAA"

        # Request only residues 2-5 (0-indexed: 2, 3, 4)
        resid_ranges = [(2, 5)]

        query_idx, ref_idx = get_matching_backbone_indices(
            query, reference, resid_ranges=resid_ranges
        )

        # Should have 3 residues * 4 atoms = 12 indices
        assert len(query_idx) == 12
        assert len(ref_idx) == 12

    def test_none_ranges_returns_all(self):
        """Verify None resid_ranges returns all residues."""
        query = MagicMock()
        query.atom_array = create_mock_atom_array(5)
        query.sequence = "AAAAA"

        reference = MagicMock()
        reference.atom_array = create_mock_atom_array(5)
        reference.sequence = "AAAAA"

        query_idx, ref_idx = get_matching_backbone_indices(
            query, reference, resid_ranges=None
        )

        # Should have all 5 residues * 4 atoms = 20 indices
        assert len(query_idx) == 20
        assert len(ref_idx) == 20

    def test_multiple_ranges(self):
        """Verify multiple non-contiguous ranges work."""
        query = MagicMock()
        query.atom_array = create_mock_atom_array(10)
        query.sequence = "AAAAAAAAAA"

        reference = MagicMock()
        reference.atom_array = create_mock_atom_array(10)
        reference.sequence = "AAAAAAAAAA"

        # Request residues 1-3 and 7-9
        resid_ranges = [(1, 3), (7, 9)]

        query_idx, ref_idx = get_matching_backbone_indices(
            query, reference, resid_ranges=resid_ranges
        )

        # Should have 4 residues * 4 atoms = 16 indices
        assert len(query_idx) == 16
        assert len(ref_idx) == 16
