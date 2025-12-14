import json
import pandas as pd
from pathlib import Path
import warnings


def parse_dataset_info(filename: str) -> dict:
    """
    Parse filenames of the form:
    external__DATASET__REPRESENTATION__MODEL.csv

    MODEL may contain underscores (e.g. SVR_rbf)
    """
    if not isinstance(filename, str):
        return {'Dataset': None, 'Representation': None, 'Model': None}

    name = Path(filename).stem
    parts = name.split('__')

    if len(parts) != 4:
        warnings.warn(f"Unexpected filename format: {filename}")
        return {'Dataset': None, 'Representation': None, 'Model': None}

    _, dataset, representation, model = parts

    return {
        'Dataset': dataset,
        'Representation': representation,
        'Model': model
    }


def process_outlier_summary(json_path: str | Path):
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    # =============================
    # Load JSON
    # =============================
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of records")

    required_keys = {'dataset', 'num_outliers'}
    for i, row in enumerate(data):
        if not required_keys.issubset(row):
            raise KeyError(f"Missing keys in entry {i}: {row}")

    # =============================
    # Build DataFrame
    # =============================
    df_raw = pd.DataFrame(data)

    parsed_df = df_raw['dataset'].apply(parse_dataset_info).apply(pd.Series)

    df = pd.concat(
        [
            parsed_df,
            df_raw['num_outliers'].rename('Outliers')
        ],
        axis=1
    )

    df = df.dropna(subset=['Dataset', 'Representation', 'Model'])

    # =============================
    # Table 1: Dataset × Model
    # =============================
    table_dataset_model = (
        df.groupby(['Dataset', 'Model'], observed=True)['Outliers']
        .mean()
        .reset_index(name='Avg_Outliers')
        .sort_values(['Dataset', 'Model'])
    )

    # =============================
    # Table 2: Representation × Model
    # =============================
    table_rep_model = (
        df.groupby(['Representation', 'Model'], observed=True)['Outliers']
        .mean()
        .reset_index(name='Avg_Outliers')
        .sort_values(['Representation', 'Model'])
    )

    # =============================
    # Save CSV files
    # =============================
    out_dir = json_path.parent

    dataset_model_csv = out_dir / "avg_outliers_dataset_model.csv"
    rep_model_csv = out_dir / "avg_outliers_representation_model.csv"

    table_dataset_model.to_csv(dataset_model_csv, index=False)
    table_rep_model.to_csv(rep_model_csv, index=False)

    # =============================
    # Console output
    # =============================
    print("\n=== Average Outliers per Dataset and Model ===")
    print(table_dataset_model.to_markdown(index=False, floatfmt=".2f"))

    print("\n=== Average Outliers per Representation and Model ===")
    print(table_rep_model.to_markdown(index=False, floatfmt=".2f"))

    print("\nModels detected:")
    print(sorted(df['Model'].unique()))

    print("\nSaved files:")
    print(dataset_model_csv)
    print(rep_model_csv)

    return table_dataset_model, table_rep_model


if __name__ == "__main__":
    process_outlier_summary(
        r"C:\Users\raula\Documents\QSAR\qsar_compact_out\outlier_summary.json"
    )
