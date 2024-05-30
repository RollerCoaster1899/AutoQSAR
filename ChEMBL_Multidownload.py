from chembl_download.chembl_download_cleaning_preprocessing import download_multiple_targets

# Downloading multiple ChEMBL dataaset and preprocesing 

chemical_targets = [
    'CHEMBL2842',
    'CHEMBL267',
    'CHEMBL1957'
]

download_multiple_targets(chemical_targets, "IC50")
