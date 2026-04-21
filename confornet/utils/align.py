from typing import TYPE_CHECKING

import biotite.structure as struc
import torch
from Bio import Align
from Bio.Data.IUPACData import protein_letters_3to1

if TYPE_CHECKING:
    from confornet.data import PredictionQuery, Reference

BACKBONE_ATOMS = ["N", "CA", "C", "O"]


def get_sequence(atom_array: struc.AtomArray) -> str:
    ca_mask = atom_array.atom_name == "CA"
    return "".join(
        protein_letters_3to1.get(r.capitalize(), "X")
        for r in atom_array[ca_mask].res_name
    )


def sequence_align(seq1: str, seq2: str) -> list[tuple[int, int]]:
    aligner = Align.PairwiseAligner(mode="global", open_gap_score=-0.5)
    alignment = aligner.align(seq1, seq2)[0]
    matching = []
    for (s1, e1), (s2, e2) in zip(alignment.aligned[0], alignment.aligned[1]):
        matching.extend(zip(range(s1, e1), range(s2, e2)))
    return matching


def get_backbone_indices(atom_array: struc.AtomArray, residue_idx: int) -> list[int]:
    res_starts = struc.get_residue_starts(atom_array)
    if residue_idx >= len(res_starts):
        return []
    start = res_starts[residue_idx]
    end = (
        res_starts[residue_idx + 1]
        if residue_idx + 1 < len(res_starts)
        else len(atom_array)
    )
    indices = []
    for atom_name in BACKBONE_ATOMS:
        for i in range(start, end):
            if atom_array.atom_name[i] == atom_name:
                indices.append(i)
                break
    return indices


def get_backbone_indices_for_ranges(
    atom_array: struc.AtomArray,
    resid_ranges: list[tuple[int, int]],
) -> list[int]:
    """Get backbone atom indices for residues in the given ranges."""
    seq_indices_set = set()
    for start, end in resid_ranges:
        seq_indices_set.update(range(start, end))
    seq_indices = sorted(seq_indices_set)

    indices = []
    for res_idx in seq_indices:
        bb = get_backbone_indices(atom_array, res_idx)
        if len(bb) == 4:
            indices.extend(bb)
    return indices


def get_matching_backbone_indices(
    query: "PredictionQuery",
    reference: "Reference",
    resid_ranges: list[tuple[int, int]] = None,
    query_resid_ranges: list[tuple[int, int]] = None,
) -> tuple[list[int], list[int]]:
    """Get matching backbone atom indices between query and reference."""
    if query_resid_ranges is not None:
        # Trim reference to resid_ranges span to avoid terminal misalignment only
        # when the query is also restricted — otherwise Biopython's aligner finds
        # scattered letter-matches across the full query instead of a contiguous window.
        if resid_ranges:
            ref_start = min(s for s, _ in resid_ranges)
            ref_end = max(e for _, e in resid_ranges)
            sub_ref_seq = reference.sequence[ref_start:ref_end]
        else:
            ref_start = 0
            sub_ref_seq = reference.sequence

        q_positions = set()
        for s, e in query_resid_ranges:
            q_positions.update(range(s, e))
        q_list = sorted(q_positions)
        sub_query_seq = "".join(query.sequence[i] for i in q_list)
        raw_matching = sequence_align(sub_query_seq, sub_ref_seq)
        matching_residues = [(q_list[q], r + ref_start) for q, r in raw_matching]
    else:
        raw_matching = sequence_align(query.sequence, reference.sequence)
        matching_residues = list(raw_matching)

    if resid_ranges:
        valid_ref = set()
        for start, end in resid_ranges:
            valid_ref.update(range(start, end))
        matching_residues = [(q, r) for q, r in matching_residues if r in valid_ref]

    query_indices, ref_indices = [], []
    for q_res, r_res in matching_residues:
        q_bb = get_backbone_indices(query.atom_array, q_res)
        r_bb = get_backbone_indices(reference.atom_array, r_res)
        if len(q_bb) == 4 and len(r_bb) == 4:
            query_indices.extend(q_bb)
            ref_indices.extend(r_bb)

    return query_indices, ref_indices


def kabsch_align(mobile: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align mobile to target using Kabsch algorithm."""
    mobile_c = mobile - mobile.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)

    H = target_c.T @ mobile_c
    U, _, Vh = torch.linalg.svd(H.float())

    d = torch.sign(torch.linalg.det(U @ Vh))
    F = torch.eye(3, device=mobile.device, dtype=U.dtype)
    F[2, 2] = d
    R = U @ F @ Vh

    return (mobile_c.float() @ R.T + target.mean(dim=0, keepdim=True)).to(mobile.dtype)
