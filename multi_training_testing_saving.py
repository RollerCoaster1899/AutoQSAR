import numpy as np
import pandas as pd
from chembl_download.chembl_download_cleaning_preprocessing import retriving_downloading_cleaning_preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from solution_coding.descriptors import preprocessing_md
from solution_coding.embedding import preprocessing_we
from solution_coding.maccs import preprocessing_mk
from solution_coding.morgan import preprocessing_mf
from xgboost import XGBRegressor


def splitting_and_post_cleaning(data, df):
    y = df["pIC50"].to_numpy()

    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=1)

    # Post-cleaning
    y_train_series = pd.Series(y_train)
    invalid_rows = y_train_series.isin([np.inf, -np.inf, np.nan])

    if any(invalid_rows):
        problematic_row_index = invalid_rows[invalid_rows].index[0]
        problematic_row = X_train[problematic_row_index]
        problematic_label = y_train[problematic_row_index]

        y_train_series[invalid_rows] = 0

    y_train = np.clip(y_train_series.to_numpy(), None, np.finfo(np.float64).max)

    return X_train, X_test, y_train, y_test

## Standard 80-20% 
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Calculate RMSE from MSE
    mae = mean_absolute_error(y_test, y_pred)

    return r2, rmse, mae

## K fold validation
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
import joblib  # Import joblib for model persistence

def train_and_evaluate_model_kfold_save_best(model, X_train, X_test, y_train, y_test, n_splits=5):
    # Concatenate the training and testing sets
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    # Create a KFold cross-validator with 5 folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Custom scorer for R2 with greater_is_better=True
    r2_scorer = make_scorer(score_func=r2_score, greater_is_better=True)

    # Initialize variables to track the best model and its R2 score
    best_model = None
    best_r2 = float('-inf')

    for train_index, test_index in kf.split(X):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        # Train the model
        model.fit(X_train_fold, y_train_fold)

        # Evaluate the model on the current fold
        r2_score_fold = model.score(X_test_fold, y_test_fold)
        rmse_score_fold = np.sqrt(mean_squared_error(y_test_fold, model.predict(X_test_fold)))
        mae_score_fold = mean_absolute_error(y_test_fold, model.predict(X_test_fold))

        # If the current model has a higher R2 score, update the best model
        if r2_score_fold > best_r2:
            best_r2 = r2_score_fold
            best_model = model

    return best_model, best_r2, rmse_score_fold, mae_score_fold

# Function to train and evaluate all models on multiple datasets
def train_and_evaluate_all_models(datasets, chemical_representation, input):
    models = {
        'kNN': KNeighborsRegressor(),
        'PLS': PLSRegression(),
        'SVM': SVR(),
        'RVM': BayesianRidge(),
        'RF': RandomForestRegressor(),
        # 'GP': GaussianProcessRegressor(),
        'XGB': XGBRegressor(),
    }

    results_r2 = {}
    results_rmse = {}
    results_mae = {}

    for dataset_name in datasets:
        model_results_r2 = []
        model_results_rmse = []
        model_results_mae = []
        potency, description, reports_len, cleaned_len = retriving_downloading_cleaning_preprocessing(dataset_name, "IC50")
        
        if input == "mf":
            chem_repre = preprocessing_mf(dataset_name, "IC50", potency)
        elif input == "md":
            chem_repre = preprocessing_md(dataset_name, "IC50", potency)
        elif input == "mk":
            chem_repre = preprocessing_mk(dataset_name, "IC50", potency)
        elif input == "we":
            chem_repre = preprocessing_we(dataset_name, "IC50", potency)

        X_train, X_test, y_train, y_test =  splitting_and_post_cleaning(chem_repre, potency)
        
        for model_name, model in models.items():
            try:
                print("Training model " + model_name + " in dataset: " + dataset_name)
                # r2, rmse, mae = train_and_evaluate_model(model, X_train, X_test, y_train, y_test) #traditional
                model, r2, rmse, mae = train_and_evaluate_model_kfold_save_best(model, X_train, X_test, y_train, y_test, 10) # 10-fold validation
                joblib.dump(model, f'{dataset_name}_{model_name}.pkl')
                model_results_r2.append({'Model': model_name, 'R2': r2})
                model_results_rmse.append({'Model': model_name, 'RMSE': rmse})
                model_results_mae.append({'Model': model_name, 'MAE': mae})
            
            except Exception as e:
                model_results_r2.append({'Model': model_name, 'R2': "-"})
                model_results_rmse.append({'Model': model_name, 'RMSE': "-"})
                model_results_mae.append({'Model': model_name, 'MAE': "-"})

        results_r2[dataset_name] = model_results_r2
        results_rmse[dataset_name] = model_results_rmse
        results_mae[dataset_name] = model_results_mae

    # Convert results to DataFrames
    df_r2 = pd.DataFrame({dataset: [result['R2'] for result in results] for dataset, results in results_r2.items()})
    df_rmse = pd.DataFrame({dataset: [result['RMSE'] for result in results] for dataset, results in results_rmse.items()})
    df_mae = pd.DataFrame({dataset: [result['MAE'] for result in results] for dataset, results in results_mae.items()})

    # Save DataFrames to CSV files
    r2_name = "r2_" + chemical_representation + ".csv"
    df_r2.to_csv(r2_name, index=False)
    rmse_name = "rmse_" + chemical_representation + ".csv"
    df_rmse.to_csv(rmse_name, index=False)
    mae_name = "mae_" + chemical_representation + ".csv"
    df_mae.to_csv(mae_name, index=False)

chemical_targets = [
    'CHEMBL2842',
    'CHEMBL267',
    'CHEMBL1957'
]

# Training and evaluating and saving the models with ECFP4
train_and_evaluate_all_models(chemical_targets, "ECPF4", "mf")

# Training and evaluating and saving  the models with ADME 
train_and_evaluate_all_models(chemical_targets, "ADME", "md")

# Training and evaluating and saving  the models with MACCS
train_and_evaluate_all_models(chemical_targets, "MACCS", "mk")

# Training and evaluating and saving the models with Word2Vec
train_and_evaluate_all_models(chemical_targets, "Word2vec", "we")
