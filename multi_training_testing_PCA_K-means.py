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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
def train_and_evaluate_model_kfold(model, X_train, X_test, y_train, y_test, n_splits=5):
    # Concatenate the training and testing sets
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    # Create a KFold cross-validator with 5 folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Use cross_val_score to perform cross-validation
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)
    rmse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
    mae_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=kf)

    # Calculate the mean scores for each metric
    r2 = np.mean(r2_scores)
    rmse = np.sqrt(-np.mean(rmse_scores))
    mae = -np.mean(mae_scores)

    return r2, rmse, mae

# Function to perform PCA and find the optimal number of components
def find_optimal_pca_components(X_train):
    pca = PCA()
    pca.fit(X_train)

    # Use the elbow technique to find the optimal number of components
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1  # 95% variance explained
    print("Optimal number of components:", optimal_components)

    return optimal_components

# Function to perform PCA on the input data
def apply_pca(X_train, X_test, n_components):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

# Function to perform KMeans and find the optimal number of clusters using the elbow technique
def find_optimal_kmeans_clusters(X_train):
    distortions = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_train)
        distortions.append(kmeans.inertia_)

    # Use the elbow technique to find the optimal number of clusters
    optimal_clusters = np.argmin(np.diff(distortions)) + 1
    print("Optimal number of clusters:", optimal_clusters)

    return optimal_clusters

# Update your train_and_evaluate_all_models function
def train_and_evaluate_all_models(datasets, chemical_representation, input):
    models = {
        'kNN': KNeighborsRegressor(),
        'PLS': PLSRegression(),
        'SVM': SVR(),
        'RVM': BayesianRidge(),
        'RF': RandomForestRegressor(),
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

        X_train, X_test, y_train, y_test = splitting_and_post_cleaning(chem_repre, potency)

        # Apply PCA
        n_components = find_optimal_pca_components(X_train)
        X_train, X_test = apply_pca(X_train, X_test, n_components)

        # Apply KMeans on PCA-transformed data
        optimal_clusters = find_optimal_kmeans_clusters(X_train)

        for model_name, model in models.items():
            try:
                print("Training model " + model_name + " in dataset: " + dataset_name)
                # Modify your model training and evaluation code here with the PCA-transformed and KMeans-clustered data
                r2, rmse, mae = train_and_evaluate_model_kfold(model, X_train, X_test, y_train, y_test, 10)  # 10-fold validation
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
    'CHEMBL4005',
    'CHEMBL3130',
    'CHEMBL3267',

    'CHEMBL3145',
    'CHEMBL4282',
    'CHEMBL2842',
    'CHEMBL3650',
    'CHEMBL2742',

    'CHEMBL1871',
    'CHEMBL203',
    'CHEMBL1957', 
    'CHEMBL279',
    'CHEMBL267',

    'CHEMBL4722',
    'CHEMBL2185',
    'CHEMBL325',
    'CHEMBL1865', 
    'CHEMBL4630'
]

# Training and evaluating the models with ECFP4
train_and_evaluate_all_models(chemical_targets, "ECPF4", "mf")

# Training and evaluating the models with ADME 
train_and_evaluate_all_models(chemical_targets, "ADME", "md")

# Training and evaluating the models with MACCS
train_and_evaluate_all_models(chemical_targets, "MACCS", "mk")

# Training and evaluating the models with Word2Vec
train_and_evaluate_all_models(chemical_targets, "Word2vec", "we")