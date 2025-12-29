import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def z_score_normalize(data, column):
    """
    Normalize data using Z-score normalization.
    """
    mean = data[column].mean()
    std = data[column].std()
    data[column + '_normalized'] = (data[column] - mean) / std
    return data

def min_max_normalize(data, column):
    """
    Normalize data using Min-Max scaling to range [0, 1].
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_scaled'] = (data[column] - min_val) / (max_val - min_val)
    return data

def clean_dataset(df, numeric_columns, outlier_factor=1.5, normalize_method='zscore'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_factor)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if normalize_method == 'zscore':
                cleaned_df = z_score_normalize(cleaned_df, col)
            elif normalize_method == 'minmax':
                cleaned_df = min_max_normalize(cleaned_df, col)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    })
    
    print("Original dataset shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['feature1', 'feature2'], normalize_method='zscore')
    print("Cleaned dataset shape:", cleaned.shape)
    print("Cleaned dataset columns:", cleaned.columns.tolist())import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. 
                        Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            elif fill_missing == 'mode':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names. Default is None.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'x', 'z', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {message}")