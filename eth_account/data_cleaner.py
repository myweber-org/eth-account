import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Strategy to fill missing values. 
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def remove_outliers_zscore(df, columns, threshold=3):
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            z_scores = np.abs(stats.zscore(cleaned_df[col]))
            cleaned_df = cleaned_df[z_scores < threshold]
    return cleaned_df

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def normalize_zscore(df, columns):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            if std_val > 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    return normalized_df

def handle_missing_values(df, strategy='mean', columns=None):
    processed_df = df.copy()
    if columns is None:
        columns = processed_df.columns
    
    for col in columns:
        if col in processed_df.columns:
            if strategy == 'mean':
                processed_df[col].fillna(processed_df[col].mean(), inplace=True)
            elif strategy == 'median':
                processed_df[col].fillna(processed_df[col].median(), inplace=True)
            elif strategy == 'mode':
                processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                processed_df = processed_df.dropna(subset=[col])
    
    return processed_df

def clean_dataset(df, numerical_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    print(f"Original dataset shape: {df.shape}")
    
    if outlier_method == 'iqr':
        df_cleaned = remove_outliers_iqr(df, numerical_columns)
    elif outlier_method == 'zscore':
        df_cleaned = remove_outliers_zscore(df, numerical_columns)
    else:
        df_cleaned = df.copy()
    
    print(f"After outlier removal: {df_cleaned.shape}")
    
    df_cleaned = handle_missing_values(df_cleaned, strategy=missing_strategy, columns=numerical_columns)
    
    if normalize_method == 'minmax':
        df_normalized = normalize_minmax(df_cleaned, numerical_columns)
    elif normalize_method == 'zscore':
        df_normalized = normalize_zscore(df_cleaned, numerical_columns)
    else:
        df_normalized = df_cleaned.copy()
    
    print(f"Final dataset shape: {df_normalized.shape}")
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000],
        'feature3': [100, 200, 300, 400, 500, 600, 700, 800, 900, 10000],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    numerical_cols = ['feature1', 'feature2', 'feature3']
    
    cleaned_df = clean_dataset(
        df, 
        numerical_columns=numerical_cols,
        outlier_method='iqr',
        normalize_method='minmax',
        missing_strategy='mean'
    )
    
    print("\nCleaned Data Summary:")
    print(cleaned_df.describe())