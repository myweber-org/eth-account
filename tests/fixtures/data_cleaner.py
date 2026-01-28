import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """
    Load a CSV file and perform basic data cleaning.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Remove duplicate rows
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - df.shape[0]
    print(f"Removed {duplicates_removed} duplicate rows.")

    # Handle missing values: fill numeric columns with median, drop rows for categorical if too many missing
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in {col} with median: {median_val}")

    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            if missing_count / len(df) < 0.1:  # Less than 10% missing
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Filled missing values in {col} with mode: {mode_val}")
            else:
                df.dropna(subset=[col], inplace=True)
                print(f"Dropped rows with missing values in {col}")

    # Remove outliers using Z-score for numeric columns (optional, based on threshold)
    z_threshold = 3
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        outlier_indices = np.where(z_scores > z_threshold)[0]
        if len(outlier_indices) > 0:
            df = df.drop(df.index[outlier_indices])
            print(f"Removed {len(outlier_indices)} outliers from {col} based on Z-score > {z_threshold}")

    # Normalize numeric columns to range [0, 1] (optional)
    for col in numeric_cols:
        if df[col].max() - df[col].min() > 0:  # Avoid division by zero
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            print(f"Normalized column {col} to range [0, 1]")

    print(f"Final data shape: {df.shape}")
    return df

def save_cleaned_data(df, output_filepath):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    if df is not None:
        df.to_csv(output_filepath, index=False)
        print(f"Cleaned data saved to {output_filepath}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"

    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    strategy (str): Imputation strategy ('mean', 'median', or 'drop').
    
    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        cleaned_df = df.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        cleaned_df = df.copy()
        for col in numeric_cols:
            cleaned_df[col].fillna(df[col].mean(), inplace=True)
    elif strategy == 'median':
        cleaned_df = df.copy()
        for col in numeric_cols:
            cleaned_df[col].fillna(df[col].median(), inplace=True)
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    return cleaned_df

def standardize_columns(df, columns=None):
    """
    Standardize specified columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to standardize. If None, all numeric columns are used.
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    standardized_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                standardized_df[col] = (df[col] - mean_val) / std_val
    
    return standardized_df