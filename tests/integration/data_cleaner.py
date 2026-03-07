
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns indices of outliers.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers.index.tolist()

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns cleaned dataframe.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column values to range [0, 1] using Min-Max scaling.
    Returns normalized series.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_data(data, column):
    """
    Standardize column values to have mean=0 and std=1.
    Returns standardized series.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    Returns processed dataframe.
    """
    if strategy == 'drop':
        return data.dropna()
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            data[col] = data[col].fillna(fill_value)
    
    return data

def clean_dataset(data, config):
    """
    Main data cleaning pipeline.
    config: dictionary with cleaning parameters
    Returns cleaned dataframe.
    """
    cleaned_data = data.copy()
    
    if config.get('handle_missing'):
        cleaned_data = handle_missing_values(
            cleaned_data, 
            strategy=config.get('missing_strategy', 'mean')
        )
    
    if config.get('remove_outliers'):
        outlier_cols = config.get('outlier_columns', cleaned_data.select_dtypes(include=[np.number]).columns)
        for col in outlier_cols:
            if col in cleaned_data.columns:
                outliers = detect_outliers_iqr(cleaned_data, col)
                cleaned_data = cleaned_data.drop(outliers)
    
    if config.get('normalize'):
        norm_cols = config.get('normalize_columns', [])
        for col in norm_cols:
            if col in cleaned_data.columns:
                cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
    
    if config.get('standardize'):
        std_cols = config.get('standardize_columns', [])
        for col in std_cols:
            if col in cleaned_data.columns:
                cleaned_data[f'{col}_standardized'] = standardize_data(cleaned_data, col)
    
    return cleaned_data
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values: 'mean', 'median', 'mode', or 'drop'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}.")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
        print("Filled missing categorical values with mode.")
    
    print(f"Cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', None, 'z']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'])
    print(f"\nData valid: {is_valid}")