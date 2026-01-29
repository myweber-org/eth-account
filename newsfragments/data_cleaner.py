import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='zscore'):
    """
    Clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    report = {
        'original_rows': len(df),
        'removed_outliers': 0,
        'normalized_columns': []
    }
    
    for column in numeric_columns:
        if column not in df.columns:
            continue
            
        if outlier_method == 'iqr':
            temp_df, removed = remove_outliers_iqr(cleaned_df, column)
        elif outlier_method == 'zscore':
            temp_df, removed = remove_outliers_zscore(cleaned_df, column)
        else:
            temp_df = cleaned_df
            removed = 0
        
        report['removed_outliers'] += removed
        cleaned_df = temp_df
        
        if normalize_method == 'minmax':
            cleaned_df[f'{column}_normalized'] = normalize_minmax(cleaned_df, column)
        elif normalize_method == 'zscore':
            cleaned_df[f'{column}_normalized'] = normalize_zscore(cleaned_df, column)
        
        report['normalized_columns'].append(column)
    
    report['final_rows'] = len(cleaned_df)
    report['removed_percentage'] = (report['removed_outliers'] / report['original_rows']) * 100
    
    return cleaned_df, report

def validate_data(df, required_columns=None, allow_nan_columns=None):
    """
    Validate data structure and content
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'empty_columns': [],
        'high_nan_columns': [],
        'warnings': []
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_result['missing_columns'] = missing
            validation_result['is_valid'] = False
    
    for column in df.columns:
        if df[column].isnull().all():
            validation_result['empty_columns'].append(column)
            validation_result['is_valid'] = False
        
        nan_percentage = df[column].isnull().mean() * 100
        if nan_percentage > 30 and (allow_nan_columns is None or column not in allow_nan_columns):
            validation_result['high_nan_columns'].append((column, nan_percentage))
            validation_result['warnings'].append(f"Column '{column}' has {nan_percentage:.1f}% NaN values")
    
    if len(df) < 10:
        validation_result['warnings'].append("Dataset has very few rows (< 10)")
    
    return validation_result

def example_usage():
    """
    Example usage of the data cleaning utilities
    """
    np.random.seed(42)
    
    sample_data = {
        'id': range(100),
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    df.loc[10, 'feature_a'] = 500
    df.loc[20, 'feature_a'] = -100
    df.loc[30:35, 'feature_b'] = np.nan
    
    print("Original dataset shape:", df.shape)
    
    cleaned_df, report = clean_dataset(
        df, 
        numeric_columns=['feature_a', 'feature_b'],
        outlier_method='iqr',
        normalize_method='zscore'
    )
    
    print("\nCleaning report:")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    validation = validate_data(cleaned_df, required_columns=['feature_a', 'feature_b'])
    print("\nValidation result:")
    print(f"Is valid: {validation['is_valid']}")
    if validation['warnings']:
        print("Warnings:", validation['warnings'])
    
    return cleaned_df

if __name__ == "__main__":
    result_df = example_usage()
    print(f"\nFinal cleaned dataset shape: {result_df.shape}")
    print("\nFirst few rows of cleaned data:")
    print(result_df.head())
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. Can be 'mean', 
                                   'median', 'mode', or a dictionary of column:value pairs.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
        else:
            cleaned_df = cleaned_df.fillna(fill_missing)
    
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} missing values remain in the dataset.")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 10, 40, 50],
        'C': ['x', 'y', 'x', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (drop duplicates, fill with mean for numeric, mode for categorical):")
    cleaned = clean_dataset(df, fill_missing={'A': 'mean', 'B': 'mean', 'C': 'mode'})
    print(cleaned)