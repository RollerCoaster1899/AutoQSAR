==============================================================================
QSAR PIPELINE - QUANTITATIVE STRUCTURE-ACTIVITY RELATIONSHIP ANALYSIS
==============================================================================

A comprehensive machine learning pipeline for predicting molecular activity 
(pIC50) across multiple ChEMBL targets using diverse chemical representations 
and state-of-the-art regression models.

==============================================================================
1. OVERVIEW
==============================================================================

This project implements an end-to-end QSAR workflow designed for scientific 
rigor and reproducibility. The pipeline performs the following steps:

1.  Data Acquisition: Downloads bioactivity data for 20 target proteins 
    directly from ChEMBL.
2.  Feature Engineering: Calculates molecular representations using RDKit, 
    Mordred, and Mol2Vec.
3.  Model Training: Trains 16 different regression models using scaffold-based 
    cross-validation.
4.  Statistical Analysis: Performs Friedman tests and Nemenyi post-hoc 
    comparisons to rank models.
5.  Prediction: Generates predictions on external test sets with full metrics.

==============================================================================
2. PROJECT STRUCTURE
==============================================================================

QSAR/
|-- README.txt                         [This file]
|-- new_training_and_testing.ipynb     [Main Jupyter notebook]
|-- correlation_filter.py              [Feature selection utility]
|
|-- CHEMBL1824/                        [Target Specific Folder]
|   |-- chembl_data/
|       |-- CHEMBL*_raw_IC50.csv       [Raw downloaded data]
|       |-- CHEMBL*_pIC50_IC50.csv     [Processed pIC50 values]
|       |-- CHEMBL*_pIC50_ECFP4.csv    [Morgan fingerprints]
|       |-- CHEMBL*_pIC50_Mordred.csv  [Mordred descriptors]
|       |-- ... (15 representations total)
|
|-- qsar_compact_out/                  [Output Directory]
|   |-- external__CHEMBL*__*.csv       [Predictions on external test sets]
|   |-- models/                        [Trained model artifacts (.joblib/.zip)]
|   |-- summary_results_external.csv   [Aggregated performance metrics]
|   |-- failures.log                   [Error log]
|   |-- analysis_reports/
|       |-- model_performance.csv
|       |-- nemenyi_pvalues.csv
|       |-- critical_difference.png    [Statistical visualization]
|
|-- catboost_info/                     [Training logs]

==============================================================================
3. KEY FEATURES
==============================================================================

[ Data Processing ]
- ChEMBL Integration: Automatic API download of IC50 data.
- Standardization: Salt stripping, canonical SMILES, duplicate aggregation.
- Transformation: pIC50 = -log10(nM).

[ Molecular Representations (12 Types) ]
- Fingerprints: ECFP4, ECFP6, MACCS, Avalon, RDKit, AtomPairs, Torsion.
- Descriptors: RDKit Extended, Physicochemical, Mordred 2D.
- Embeddings: Mol2Vec (Word2Vec).

[ Machine Learning Models (16 Algorithms) ]
- Linear: Ridge, Lasso, ElasticNet, BayesianRidge.
- SVM: SVR (RBF kernel), SVR (linear kernel).
- Ensembles: RandomForest, ExtraTrees, HistGradientBoosting, XGBoost.
- Other: KNN, PLS, MLP (Neural Net).

[ Validation Strategy ]
- Scaffold-based Splitting: Ensures chemically distinct test sets.
- Group K-Fold CV: Respects molecular scaffolds during tuning.
- Hyperparameter Tuning: Random search and Halving search (20 iterations).
- Statistical Tests: Friedman Test and Nemenyi Critical Difference plots.

==============================================================================
4. INSTALLATION
==============================================================================

Requirements: Python 3.9+

Step 1: Create Environment
--------------------------
conda create -n qsar python=3.11
conda activate qsar

Step 2: Install Dependencies
----------------------------
pip install pandas numpy rdkit chembl_webresource_client mordred mol2vec gensim tqdm
pip install scikit-learn xgboost jupyter matplotlib scipy scikit-posthocs joblib

Step 3: Path Configuration
--------------------------
Open "new_training_and_testing.ipynb" and edit the BASE_DIR variable in the
QSARRunner class to match your local path.

==============================================================================
5. USAGE
==============================================================================

Quick Start
-----------
Run the Jupyter Notebook:
> jupyter notebook new_training_and_testing.ipynb

Execute the cells in sequence:
1. Imports and Dependencies
2. Configuration
3. Feature Calculation
4. Model Training (Main Pipeline)
5. Analysis and Plotting

Running Programmatically
------------------------
Inside the notebook or a Python script:

    from new_training_and_testing import QSARRunner
    
    # Initialize and run the full pipeline
    runner = QSARRunner()
    runner.run()

Running Specific Components
---------------------------
    # Download data only
    process_chembl_data("CHEMBL1824")

    # Calculate features only
    calculate_representations("CHEMBL1824")

==============================================================================
6. CONFIGURATION OPTIONS
==============================================================================

Adjust these constants within the QSARRunner class:

FAST_MODE = True           # Skip expensive feature selection for speed
USE_HALVING_SEARCH = True  # Use HalvingRandomSearchCV (faster)
USE_GPU = False            # Enable GPU for XGBoost
COMPRESS_MODELS = True     # Save models as .zip to save space
RUN_OOF_EVALUATION = False # Skip internal validation metrics

OUTER_FOLDS = 5            # Number of cross-validation folds
HOLDOUT_FRAC = 0.15        # Fraction of data for external test set
N_ITER_SEARCH = 20         # Hyperparameter search iterations

==============================================================================
7. OUTPUT FILES AND METRICS
==============================================================================

Predictions
-----------
File: external__CHEMBL*__REPRESENTATION__MODEL.csv
Columns: 
- molecule_chembl_id
- std_canonical_smiles
- pIC50 (Observed)
- pred (Predicted)

Summary Results
---------------
File: summary_results_external.csv
Contains the R-squared (R2) and RMSE scores for every model/dataset pair.

Analysis Reports
----------------
File: critical_difference.png
A visual ranking of models. Models connected by a horizontal bar are not 
statistically different (p > 0.05).
