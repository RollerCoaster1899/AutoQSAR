import pandas as pd
from pathlib import Path
import numpy as np

# Change this to your results folder
target_dir = Path("./qsar_compact_out") 

# Collect all CSV files in the target directory
files = list(target_dir.glob("external__*.csv"))

print(f"Checking {len(files)} files for outliers...")

# Loop over each file and process them
for f in files:
    try:
        # Load the dataset into a pandas DataFrame
        df = pd.read_csv(f)
        
        # Check if 'pred' column exists
        if 'pred' not in df.columns: 
            continue
        
        # Calculate Q1 (25th percentile), Q3 (75th percentile), and IQR
        Q1 = df['pred'].quantile(0.25)
        Q3 = df['pred'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define the outlier thresholds with the factor of 1.35
        lower_bound = Q1 - 1.35 * IQR
        upper_bound = Q3 + 1.35 * IQR
        
        # Print outliers detection criteria
        print(f"\nOutlier detection criteria for {f.name}:")
        print(f"   Lower bound: {lower_bound:.2f}")
        print(f"   Upper bound: {upper_bound:.2f}")
        
        # Identify outliers (pred values outside the calculated bounds)
        outliers = df[(df['pred'] < lower_bound) | (df['pred'] > upper_bound)]
        
        if not outliers.empty:
            print(f"⚠️ FOUND OUTLIERS in: {f.name}")
            print(f"   Number of outliers: {len(outliers)} out of {len(df)}")
            print("   Example outlier row:")
            print(outliers.head(1))
            
            # Drop the outliers (rows where 'pred' is outside of the IQR range)
            df_cleaned = df[~df.index.isin(outliers.index)]
            print(f"   Cleaned dataset: {len(df_cleaned)} rows after removing outliers.")
            
            # Save the cleaned dataset, overwriting the original file
            df_cleaned.to_csv(f, index=False)
        
        else:
            print(f"   No outliers detected in {f.name}.")
        
    except Exception as e:
        print(f"⚠️ Error processing {f.name}: {e}")

print("\nAll files processed successfully.")
