"""Molecule cleaning + descriptor generation (ECFP fingerprints, RDKit2D
physicochemical descriptors), shared by every benchmark backend/variant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from chembl_structure_pipeline.standardizer import standardize_mol as chembl_standardizer
from rdkit import Chem
from rdkit.Chem import DataStructs, Descriptors, rdFingerprintGenerator


def mol_from_smiles(smiles, *args, standardizer=chembl_standardizer) -> Chem.Mol:
    """Convert a SMILES string to an RDKit Mol object.

    :param smiles: SMILES string. Required.
    :param standardizer: function to standardize the molecule. Defaults to
        ChEMBL standardizer. Use None to skip.
    :return: RDKit Mol object, or None if the SMILES could not be parsed.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if not mol:
        return None
    if standardizer is None:
        return mol
    return standardizer(mol, check_exclusion=True, sanitize=True)


def fp_to_np(fp):
    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def gen_ecfp_arr(mol_df, mol_col, fp_size=1024, fp_radius=2, n_threads=-1):
    """Morgan (ECFP)-style fingerprints as a binary uint32 array."""
    mols = list(mol_df[mol_col])
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_size)
    fps = fpg.GetFingerprints(mols, numThreads=n_threads)
    fps_np = np.array([fp_to_np(i) for i in fps], dtype=np.uint32)
    return fps_np


def gen_rdkit2D_arr(mol_df, mol_col):
    """All RDKit 2D physicochemical descriptors as a continuous array (must
    be binarized, e.g. via `tm_qsar_benchmark.binarizer.Binarizer`, before
    use with a Tsetlin Machine)."""
    mols = list(mol_df[mol_col])
    descrs = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
    desc_np = pd.DataFrame(descrs).values
    return desc_np
