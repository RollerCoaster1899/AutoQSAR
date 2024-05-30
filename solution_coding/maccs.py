import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import MACCSkeys, Descriptors

def generate_maccs_keys(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            maccs_keys = MACCSkeys.GenMACCSKeys(mol)
            maccs_keys_list = list(map(int, maccs_keys.ToBitString()))
            return np.array(maccs_keys_list)
        else:
            return None
    except Exception as e:
        print(f"An unexpected error occurred in generate_maccs_keys_df function: {e}")
        return None

def preprocessing_mk(target, assay_type, data):
    try:
        print("Starting molecules fingerprinting: ")
        maccs_keys_df = data["canonical_smiles"].apply(generate_maccs_keys).dropna().tolist()
        return np.array(maccs_keys_df)
    except Exception as e:
        print(f"An unexpected error occurred in df word_embeddings function: {e}")
        return None