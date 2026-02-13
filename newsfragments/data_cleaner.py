
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a dataset by handling missing values and removing outliers.
    
    Args:
        df (pd.DataFrame): Input dataframe to clean
        missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
        outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Remove outliers using Z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / cleaned_df[numeric_cols].std())
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    cleaned_df = cleaned_df[outlier_mask]
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=10):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"Dataset has less than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        if np.any(np.isinf(df[numeric_cols].values)):
            return False, "Dataset contains infinite values"
    
    return True, "Dataset is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = pd.DataFrame({
#         'A': [1, 2, np.nan, 4, 100],
#         'B': [5, 6, 7, np.nan, 8],
#         'C': [9, 10, 11, 12, 13]
#     })
#     
#     cleaned = clean_dataset(sample_data, missing_strategy='mean')
#     print("Original shape:", sample_data.shape)
#     print("Cleaned shape:", cleaned.shape)
#     
#     is_valid, message = validate_data(cleaned, required_columns=['A', 'B', 'C'])
#     print(f"Validation: {is_valid} - {message}")
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
        df_clean = df_clean[(z_scores < threshold) | (df_clean[col].isna())]
    return df_clean

def normalize_minmax(df, columns):
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        if max_val != min_val:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0
    return df_normalized

def normalize_zscore(df, columns):
    df_normalized = df.copy()
    for col in columns:
        mean_val = df_normalized[col].mean()
        std_val = df_normalized[col].std()
        if std_val != 0:
            df_normalized[col] = (df_normalized[col] - mean_val) / std_val
        else:
            df_normalized[col] = 0
    return df_normalized

def handle_missing_values(df, strategy='mean', columns=None):
    df_filled = df.copy()
    if columns is None:
        columns = df_filled.columns
    
    for col in columns:
        if df_filled[col].dtype in ['float64', 'int64']:
            if strategy == 'mean':
                fill_value = df_filled[col].mean()
            elif strategy == 'median':
                fill_value = df_filled[col].median()
            elif strategy == 'mode':
                fill_value = df_filled[col].mode()[0]
            else:
                fill_value = 0
            df_filled[col].fillna(fill_value, inplace=True)
        else:
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
    
    return df_filled

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    df_clean = df.copy()
    
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df_clean, numeric_columns)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df_clean, numeric_columns)
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy, columns=numeric_columns)
    
    if normalize_method == 'minmax':
        df_clean = normalize_minmax(df_clean, numeric_columns)
    elif normalize_method == 'zscore':
        df_clean = normalize_zscore(df_clean, numeric_columns)
    
    return df_clean