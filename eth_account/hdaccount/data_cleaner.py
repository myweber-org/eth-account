
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns):
    """Normalize data using min-max scaling."""
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def handle_missing_values(df, strategy='mean'):
    """Handle missing values with specified strategy."""
    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].dtype in ['float64', 'int64']:
            if strategy == 'mean':
                df_filled[col].fillna(df_filled[col].mean(), inplace=True)
            elif strategy == 'median':
                df_filled[col].fillna(df_filled[col].median(), inplace=True)
            elif strategy == 'mode':
                df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
    return df_filled

def save_cleaned_data(df, filepath):
    """Save cleaned dataset to CSV."""
    df.to_csv(filepath, index=False)

def main():
    # Example usage
    input_file = 'raw_data.csv'
    output_file = 'cleaned_data.csv'
    
    # Load data
    data = load_dataset(input_file)
    print(f"Original data shape: {data.shape}")
    
    # Handle missing values
    data = handle_missing_values(data, strategy='mean')
    
    # Select numeric columns for cleaning
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove outliers
    data = remove_outliers_iqr(data, numeric_cols)
    print(f"Data shape after outlier removal: {data.shape}")
    
    # Normalize data
    data = normalize_data(data, numeric_cols)
    
    # Save cleaned data
    save_cleaned_data(data, output_file)
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    main()