import os
import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
 
def target_retrieval(search):
    try:
        target = new_client.target
        target_query = target.search(search)
        targets = pd.DataFrame.from_dict(target_query)
        return targets
    except Exception as e:
        print(f"Error in target retrieval: {e}")
        return None

def target_info_retrieval(targets, target_chembl_id, assay_type):
    try:
        target = targets["target_chembl_id"].iloc[0]
        activity = new_client.activity
        res = activity.filter(target_chembl_id=target).filter(standard_type=assay_type)
        reports = pd.DataFrame.from_dict(res)
        description = reports["target_pref_name"].iloc[0]
        uniprot = reports["target_pref_name"].iloc[0]
        return reports, description
    except Exception as e:
        print(f"Error in target info retrieval: {e}")
        return None

def clean(reports, target_chembl_id, assay_type):
    try:
        cleaned_1 = reports.dropna(subset=["standard_value","canonical_smiles"])
        cleaned_2 = cleaned_1.drop_duplicates(subset=["canonical_smiles"])
        cleaned_combined = cleaned_2[["molecule_chembl_id","standard_value","canonical_smiles"]]
        return cleaned_combined
    except Exception as e:
        print(f"Error in cleaning: {e}")
        return None

def norm_value(df_cleaned,target_chembl_id, assay_type):
    try:
        norm = []
        dataframe = df_cleaned.reset_index(drop=True)

        for i in dataframe["standard_value"]:
            if isinstance(i, str):
                i = float(i)
            if i is not None and i > 100000000:
                i = 100000000
            if i is not None:
                norm.append(i)

        dataframe['standard_value_norm'] = norm
        x = dataframe.drop("standard_value", axis=1)
        return x
    except Exception as e:
        print(f"Error in normalization: {e}")
        return None

def Negative_log(df_normalized,target_chembl_id, assay_type):
    try:
        potencies = []

        for i in df_normalized['standard_value_norm']:
            molar = i * (10 ** -9)
            potencies.append(-np.log10(molar))
        df_normalized["pIC50"] = potencies
        return df_normalized
    except Exception as e:
        print(f"Error in negative log calculation: {e}")
        return None

def retriving_downloading_cleaning_preprocessing(target_chembl_id, assay_type):
    target_folder = os.path.join("ChEMBL_data", target_chembl_id, "chembl_data")  

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    chembl_preprocessed_data = target_chembl_id + "_pIC50_" + assay_type + ".csv"
    chembl_raw_data = target_chembl_id + "_raw_" + assay_type + ".csv"
    target_folder_raw_file = os.path.join(target_folder, chembl_raw_data)
    target_folder_pic50_file = os.path.join(target_folder, chembl_preprocessed_data)

    if os.path.isfile(target_folder_pic50_file):
        print(target_folder_pic50_file + " file already exists.")
        potency = pd.read_csv(target_folder_pic50_file)
        raw = pd.read_csv(target_folder_raw_file)
        description = raw["target_pref_name"].iloc[0]

        return potency, description, len(raw), len(potency)
    else:   
        try:
            print("retrieving target: " + target_chembl_id)
            targets = target_retrieval(target_chembl_id)
            if targets is None:
                print("Error in target retrieval. Exiting.")
                return None

            print("downloading assay info: " + target_chembl_id)
            reports, description = target_info_retrieval(targets, target_chembl_id, assay_type)
            target_folder_reports_file = os.path.join(target_folder, target_chembl_id + "_raw_" + assay_type + ".csv")
            reports.to_csv(target_folder_reports_file, index=False)

            if reports is None:
                print("Error in target info retrieval. Exiting.")
                return None

            print("cleaning assay info: " + target_chembl_id)
            cleaned = clean(reports, target_chembl_id, assay_type)
            target_folder_cleaned_file = os.path.join(target_folder, target_chembl_id + "_cleaned_" + assay_type + ".csv")
            cleaned.to_csv(target_folder_cleaned_file, index=False)

            if cleaned is None:
                print("Error in cleaning. Exiting.")
                return None

            print("Normalizing assay info: " + target_chembl_id + " cleaned length: " + str(len(cleaned)))
            normalized = norm_value(cleaned, target_chembl_id, assay_type)
            target_folder_normalized_file = os.path.join(target_folder, target_chembl_id + "_normalized_" + assay_type + ".csv")
            normalized.to_csv(target_folder_normalized_file, index=False)
            
            if normalized is None:
                print("Error in normalization. Exiting.")
                return None

            print("Calculating potency of assay info: " + target_chembl_id)
            potency = Negative_log(normalized, target_chembl_id, assay_type)
            target_folder_potency_file = os.path.join(target_folder, target_chembl_id + "_pIC50_" + assay_type + ".csv")
            potency.to_csv(target_folder_potency_file, index=False)

            if potency is None:
                print("Error in negative log calculation. Exiting.")
                return None

            reports_len = len(reports)
            cleaned_len = len(cleaned)
            return potency, description, reports_len, cleaned_len
        
        except Exception as e:
            print(f"Unexpected error: {e}")

def download_multiple_targets(targets, assay_type):
    data_columns = ['Target', "Description", 'Raw_length', 'Cleaned_len']
    DATA = pd.DataFrame(columns=data_columns)

    for target in targets:
        potency, description, reports_len, cleaned_len = retriving_downloading_cleaning_preprocessing(target, assay_type)
        DATA = DATA._append({'Target': target, "Description":description, 'Raw_length': reports_len, 'Cleaned_len': cleaned_len}, ignore_index=True)

    DATA.to_csv('targets_info.csv', index=False)