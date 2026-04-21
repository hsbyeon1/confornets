from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import torch


def load_structure(path: Path | str) -> struc.AtomArray:
    path = Path(path)
    if path.suffix.lower() == ".pdb":
        return pdb.get_structure(pdb.PDBFile.read(path), model=1)
    return pdbx.get_structure(pdbx.CIFFile.read(path), model=1)


def write_cif(
    atom_array: struc.AtomArray,
    coords: np.ndarray | torch.Tensor,
    output_path: Path | str,
    plddt: np.ndarray | None = None,
):
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().float().numpy()

    atom_array = atom_array.copy()
    atom_array.coord = coords

    if plddt is not None:
        atom_array.set_annotation("b_factor", plddt)
    else:
        atom_array.set_annotation("b_factor", np.ones(len(atom_array)) * 50.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cif_file = pdbx.CIFFile()
    pdbx.set_structure(cif_file, atom_array)
    cif_file.write(output_path)
