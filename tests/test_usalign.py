"""Tests for USAlign utilities with separate alignment and metric ranges."""

from unittest.mock import MagicMock, patch

import numpy as np

from confornet.data import Reference
from confornet.utils.usalign import (
    _compute_rmsd_from_distances,
    _get_metric_indices_in_alignment,
    _parse_do_output,
)


class TestParseDoOutput:
    """Tests for parsing USAlign -do flag output."""

    def test_basic_parsing(self):
        """Test parsing standard -do output."""
        output = """\
Some header info
###############	###############	#########
#Aligned atom 1	Aligned atom 2 	Distance#
 CA  SER A   1.	 CA  SER A   1.	    1.234
 CA  PRO A   2.	 CA  PRO A   2.	    2.345
 CA  ALA A   3.	 CA  ALA A   3.	    0.567
###############	###############	#########
Some footer
"""
        distances = _parse_do_output(output)

        assert len(distances) == 3
        assert distances[0] == (1, 1, 1.234)
        assert distances[1] == (2, 2, 2.345)
        assert distances[2] == (3, 3, 0.567)

    def test_empty_output(self):
        """Test parsing output with no distance table."""
        output = "No alignment found"
        distances = _parse_do_output(output)
        assert distances == []

    def test_different_residue_ids(self):
        """Test parsing when mobile and target have different residue IDs."""
        output = """\
###############	###############	#########
#Aligned atom 1	Aligned atom 2 	Distance#
 CA  SER A   5.	 CA  SER A  10.	    1.000
 CA  PRO A   6.	 CA  PRO A  11.	    2.000
###############	###############	#########
"""
        distances = _parse_do_output(output)

        assert len(distances) == 2
        assert distances[0] == (5, 10, 1.0)
        assert distances[1] == (6, 11, 2.0)


class TestGetMetricIndicesInAlignment:
    """Tests for mapping metric ranges to sequential CIF indices."""

    def test_full_ranges(self):
        """When alignment and metric ranges cover the full sequence, all are included."""
        full_range = [(0, 10)]
        indices = _get_metric_indices_in_alignment(full_range, full_range)
        assert indices == set(range(1, 11))

    def test_alignment_equals_metric(self):
        """When ranges match, all aligned residues are metric residues."""
        ranges = [(1, 6)]  # Residues 1-5 (1-indexed, end exclusive)
        indices = _get_metric_indices_in_alignment(ranges, ranges)
        assert indices == {1, 2, 3, 4, 5}

    def test_metric_subset_of_alignment(self):
        """Metric range is a subset of alignment range."""
        alignment_ranges = [(1, 11)]  # Residues 1-10
        metric_ranges = [(3, 8)]  # Residues 3-7

        indices = _get_metric_indices_in_alignment(alignment_ranges, metric_ranges)

        # CIF indices 3-7 map to metric residues
        assert indices == {3, 4, 5, 6, 7}

    def test_alignment_subset_metric_superset(self):
        """Alignment range is smaller than metric range."""
        # Align on residues 3-7 (0-indexed: 2-6)
        alignment_ranges = [(3, 8)]  # 1-indexed, end exclusive
        # Metric on residues 1-10
        metric_ranges = [(1, 11)]

        indices = _get_metric_indices_in_alignment(alignment_ranges, metric_ranges)

        # All 5 aligned residues (CIF ids 1-5) are in metric range
        assert indices == {1, 2, 3, 4, 5}

    def test_disjoint_ranges(self):
        """Alignment and metric ranges don't overlap."""
        alignment_ranges = [(1, 6)]  # Residues 1-5
        metric_ranges = [(6, 11)]  # Residues 6-10

        indices = _get_metric_indices_in_alignment(alignment_ranges, metric_ranges)

        # No overlap - empty set
        assert indices == set()

    def test_multiple_ranges(self):
        """Multiple alignment and metric ranges."""
        # Align on residues 1-3 and 8-10 (CIF indices will be 1-6)
        alignment_ranges = [(1, 4), (8, 11)]
        # Metric on residues 2-3 and 9-10
        metric_ranges = [(2, 4), (9, 11)]

        indices = _get_metric_indices_in_alignment(alignment_ranges, metric_ranges)

        # Alignment positions: 0,1,2 (from range1) and 7,8,9 (from range2)
        # These become CIF indices 1,2,3,4,5,6
        # Metric positions: 1,2 (from range1) and 8,9 (from range2)
        # CIF idx 1 -> orig_pos 0, not in metric
        # CIF idx 2 -> orig_pos 1, in metric
        # CIF idx 3 -> orig_pos 2, in metric
        # CIF idx 4 -> orig_pos 7, not in metric
        # CIF idx 5 -> orig_pos 8, in metric
        # CIF idx 6 -> orig_pos 9, in metric
        assert indices == {2, 3, 5, 6}


class TestComputeRmsdFromDistances:
    """Tests for computing RMSD from filtered distances."""

    def test_all_included(self):
        """RMSD when all distances are included."""
        distances = [(1, 1, 1.0), (2, 2, 2.0), (3, 3, 3.0)]
        metric_indices = {1, 2, 3}

        rmsd = _compute_rmsd_from_distances(distances, metric_indices)

        # RMSD = sqrt(mean([1^2, 2^2, 3^2])) = sqrt((1+4+9)/3) = sqrt(14/3)
        expected = np.sqrt(14 / 3)
        assert abs(rmsd - expected) < 1e-10

    def test_partial_filter(self):
        """RMSD when only some distances are included."""
        distances = [(1, 1, 1.0), (2, 2, 2.0), (3, 3, 3.0), (4, 4, 4.0)]
        metric_indices = {2, 3}  # Only include residues 2 and 3

        rmsd = _compute_rmsd_from_distances(distances, metric_indices)

        # RMSD = sqrt(mean([2^2, 3^2])) = sqrt((4+9)/2) = sqrt(6.5)
        expected = np.sqrt(6.5)
        assert abs(rmsd - expected) < 1e-10

    def test_empty_filter(self):
        """RMSD when no distances match filter."""
        distances = [(1, 1, 1.0), (2, 2, 2.0)]
        metric_indices = {5, 6}  # No matching indices

        rmsd = _compute_rmsd_from_distances(distances, metric_indices)
        assert rmsd == float("inf")

    def test_single_residue(self):
        """RMSD with single residue."""
        distances = [(1, 1, 2.5)]
        metric_indices = {1}

        rmsd = _compute_rmsd_from_distances(distances, metric_indices)
        assert rmsd == 2.5


class TestRangesMatch:
    """Tests for Reference.ranges_match field."""

    def test_ranges_match_when_equal(self):
        """ranges_match is True when both ranges are equal."""
        with patch("confornet.data.load_structure") as mock_load:
            mock_array = MagicMock()
            mock_array.coord = np.zeros((10, 3))
            mock_load.return_value = mock_array

            with patch("confornet.data.get_sequence") as mock_seq:
                mock_seq.return_value = "ACDEFGHIKL"

                ref = Reference.from_path(
                    "test",
                    MagicMock(),
                    alignment_resid_ranges=[[0, 5]],
                    metric_resid_ranges=[[0, 5]],
                )

                assert ref.ranges_match is True
                assert ref.alignment_resid_ranges == [(0, 5)]
                assert ref.metric_resid_ranges == [(0, 5)]

    def test_ranges_match_when_different(self):
        """ranges_match is False when ranges differ."""
        with patch("confornet.data.load_structure") as mock_load:
            mock_array = MagicMock()
            mock_array.coord = np.zeros((10, 3))
            mock_load.return_value = mock_array

            with patch("confornet.data.get_sequence") as mock_seq:
                mock_seq.return_value = "ACDEFGHIKL"

                ref = Reference.from_path(
                    "test",
                    MagicMock(),
                    alignment_resid_ranges=[[0, 10]],
                    metric_resid_ranges=[[2, 8]],
                )

                assert ref.ranges_match is False
                assert ref.alignment_resid_ranges == [(0, 10)]
                assert ref.metric_resid_ranges == [(2, 8)]

    def test_no_ranges_specified(self):
        """When no ranges given, defaults to full sequence."""
        with patch("confornet.data.load_structure") as mock_load:
            mock_array = MagicMock()
            mock_array.coord = np.zeros((10, 3))
            mock_load.return_value = mock_array

            with patch("confornet.data.get_sequence") as mock_seq:
                mock_seq.return_value = "ACDEFGHIKL"

                ref = Reference.from_path("test", MagicMock())

                assert ref.ranges_match is True
                assert ref.alignment_resid_ranges == [(0, 10)]
                assert ref.metric_resid_ranges == [(0, 10)]
