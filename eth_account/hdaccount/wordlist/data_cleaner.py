
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_cols
        validation_results['all_required_columns_present'] = len(missing_cols) == 0
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, None, 15.3, 20.1, None, 25.7],
        'category': ['A', 'B', 'B', 'A', 'C', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned)
    print("\nCleaned validation results:")
    print(validate_dataset(cleaned))
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    Returns filtered DataFrame and outlier indices.
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    mask = (dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)
    outliers = dataframe[~mask].index.tolist()
    
    return dataframe[mask].copy(), outliers

def z_score_normalize(dataframe, columns):
    """
    Apply z-score normalization to specified columns.
    Returns DataFrame with normalized columns.
    """
    result = dataframe.copy()
    
    for col in columns:
        if col not in result.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        mean_val = result[col].mean()
        std_val = result[col].std()
        
        if std_val > 0:
            result[f"{col}_normalized"] = (result[col] - mean_val) / std_val
        else:
            result[f"{col}_normalized"] = 0
    
    return result

def min_max_normalize(dataframe, columns, feature_range=(0, 1)):
    """
    Apply min-max normalization to specified columns.
    Returns DataFrame with normalized columns.
    """
    result = dataframe.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col not in result.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        col_min = result[col].min()
        col_max = result[col].max()
        
        if col_max > col_min:
            normalized = (result[col] - col_min) / (col_max - col_min)
            result[f"{col}_scaled"] = normalized * (max_val - min_val) + min_val
        else:
            result[f"{col}_scaled"] = min_val
    
    return result

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    result = dataframe.copy()
    
    if columns is None:
        columns = result.columns
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if result[col].isnull().any():
            if strategy == 'mean':
                fill_value = result[col].mean()
            elif strategy == 'median':
                fill_value = result[col].median()
            elif strategy == 'mode':
                fill_value = result[col].mode()[0] if not result[col].mode().empty else 0
            elif strategy == 'drop':
                result = result.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            result[col] = result[col].fillna(fill_value)
    
    return result

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    Returns boolean indicating validity and error message if invalid.
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"