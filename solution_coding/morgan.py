import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_morgan_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)  # Change the second argument to the radius you want
            morgan_fp_list = list(map(int, morgan_fp.ToBitString()))
            return np.array(morgan_fp_list)
        else:
            return None
    except Exception as e:
        print(f"An unexpected error occurred in generate_morgan_fingerprint function: {e}")
        return None

def preprocessing_mf(target, assay_type, data):
    try:
        print("Starting molecules fingerprinting: ")
        morgan_fingerprints = data["canonical_smiles"].apply(generate_morgan_fingerprint).dropna().tolist()
        return np.array(morgan_fingerprints)
    except Exception as e:
        print(f"An unexpected error occurred in preprocessing_mk function: {e}")
        return None