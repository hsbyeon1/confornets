"""Atom-pair selection from a ``guidances`` block, for the atom-pair distance diversity loss.

The block has the shape used by boltz2-guidance and the Protenix guidance fork, so a
guidance specification can be copied between the tools unchanged::

    "guidances": [
        {
            "particle": {
                "atom_sets": [
                    [["A", "79:89", "CA"], ["A", "162:167", "CA"]]
                ]
            }
        }
    ]

An *atom set* is a list of atom specifications ``[chain_id, residue, atom_name]`` where
``residue`` is a 1-based residue number or a half-open ``"start:end"`` range (``"74:87"`` is
residues 74-86). Every pair of atoms within one set becomes a pair (an all-pairs clique); the
pairs of all sets and all entries are united.

``atom_set_pairs`` (ConforNets extension) lists ``[set_a, set_b]`` couples; each contributes only
the cross pairs a x b, e.g. TM6 intracellular-end atoms against a TM1/2/4 anchor without the
anchor-anchor pairs a clique would add. An entry may carry either key or both. Residue numbers are ``atom_array.res_id``,
i.e. the numbering written to the output CIFs.
"""

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

import numpy as np
import torch

_RESIDUE_RANGE = re.compile(r"(\d+):(\d+)")


class AtomSpec(NamedTuple):
    chain_id: str
    res_id: int
    atom_name: str


def parse_atom_set(atom_set: Iterable[Any]) -> list[AtomSpec]:
    """Expand ``[chain_id, residue, atom_name]`` specifications into single atoms."""
    specs: list[AtomSpec] = []
    for atom in atom_set:
        if len(atom) != 3:
            raise ValueError(f"Invalid atom specification {atom!r}: expecting [chain_id, residue, atom_name].")
        chain_id, residue, atom_name = atom
        if isinstance(residue, bool):
            raise ValueError(f"Invalid residue: {residue!r}")
        if isinstance(residue, int):
            res_ids = [residue]
        else:
            match = _RESIDUE_RANGE.fullmatch(str(residue))
            if match is None:
                raise ValueError(f"Invalid residue range: {residue!r}")
            start, end = map(int, match.groups())
            res_ids = list(range(start, end))
        specs.extend(AtomSpec(str(chain_id), r, str(atom_name)) for r in res_ids)
    return specs


def resolve_atom_indices(atom_array, specs: Iterable[AtomSpec]) -> list[int]:
    """Indices into ``atom_array`` (first occurrence), one per spec; raises on a missing atom."""
    lookup: dict[tuple[str, int, str], int] = {}
    for index, key in enumerate(zip(atom_array.chain_id, atom_array.res_id, atom_array.atom_name)):
        lookup.setdefault((str(key[0]), int(key[1]), str(key[2])), index)
    out = []
    for spec in specs:
        index = lookup.get((spec.chain_id, spec.res_id, spec.atom_name))
        if index is None:
            chains = sorted({str(c) for c in atom_array.chain_id})
            raise ValueError(f"Atom {tuple(spec)} is not in the structure. Available chains: {chains}.")
        out.append(index)
    return out


def build_pair_index(atom_array, guidances: Iterable[Any] | None) -> torch.Tensor | None:
    """``[2, N_pair]`` long tensor of atom-index pairs from a ``guidances`` block (None if empty)."""
    if not guidances:
        return None
    if isinstance(guidances, Mapping):
        raise ValueError("'guidances' must be a list of guidance entries.")
    pairs: set[tuple[int, int]] = set()
    for entry in guidances:
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise ValueError(f"Invalid guidance entry {entry!r}: expecting a single-key mapping.")
        ((kind, values),) = entry.items()
        if kind != "particle":
            raise ValueError(f"Unsupported guidance type {kind!r}; only 'particle' atom sets are used.")
        if "atom_sets" not in values and "atom_set_pairs" not in values:
            raise ValueError("A 'particle' entry needs 'atom_sets' and/or 'atom_set_pairs'.")
        for atom_set in values.get("atom_sets", []):
            idx = resolve_atom_indices(atom_array, parse_atom_set(atom_set))
            if len(set(idx)) < 2:
                raise ValueError(f"Each atom set needs at least two atoms, got {len(set(idx))}.")
            pairs.update(combinations(sorted(set(idx)), 2))
        for couple in values.get("atom_set_pairs", []):
            if len(couple) != 2:
                raise ValueError(f"Each atom_set_pairs entry must be [set_a, set_b], got {couple!r}.")
            ia = set(resolve_atom_indices(atom_array, parse_atom_set(couple[0])))
            ib = set(resolve_atom_indices(atom_array, parse_atom_set(couple[1])))
            if not ia or not ib:
                raise ValueError("Both sets of an atom_set_pairs entry need at least one atom.")
            pairs.update((min(a, b), max(a, b)) for a in ia for b in ib if a != b)
    if not pairs:
        return None
    arr = np.asarray(sorted(pairs), dtype=np.int64)
    return torch.from_numpy(arr.T.copy())


def load_guidances(path: str | Path) -> list:
    """Read a ``guidances`` list from a JSON file.

    Accepts the bare list, a query/input dict with a ``guidances`` key, or a one-element
    list of such dicts (Protenix input format).
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], Mapping) and "guidances" in data[0]:
        data = data[0]
    if isinstance(data, Mapping):
        if "guidances" not in data:
            raise ValueError(f"{path}: no 'guidances' key")
        data = data["guidances"]
    if not isinstance(data, list):
        raise ValueError(f"{path}: 'guidances' must be a list")
    return data
