
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'value': np.random.normal(100, 15, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Add some outliers
    sample_data.loc[1000] = [500, 'A']
    sample_data.loc[1001] = [-100, 'B']
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:", calculate_summary_stats(sample_data, 'value'))
    
    cleaned_data = clean_numeric_data(sample_data, ['value'])
    print("Cleaned data shape:", cleaned_data.shape)
    print("Cleaned statistics:", calculate_summary_stats(cleaned_data, 'value'))
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method.
    Returns filtered data and outlier indices.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
    outliers = data[~mask].index.tolist()
    
    return data[mask].copy(), outliers

def normalize_minmax(data, columns=None):
    """
    Normalize specified columns using Min-Max scaling.
    If columns is None, normalize all numeric columns.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            col_min = data[col].min()
            col_max = data[col].max()
            
            if col_max != col_min:
                normalized_data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                normalized_data[col] = 0
    
    return normalized_data

def standardize_zscore(data, columns=None):
    """
    Standardize specified columns using Z-score normalization.
    If columns is None, standardize all numeric columns.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    standardized_data = data.copy()
    
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            mean_val = data[col].mean()
            std_val = data[col].std()
            
            if std_val > 0:
                standardized_data[col] = (data[col] - mean_val) / std_val
            else:
                standardized_data[col] = 0
    
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    processed_data = data.copy()
    
    if strategy == 'drop':
        return processed_data.dropna(subset=columns)
    
    for col in columns:
        if col in processed_data.columns and processed_data[col].isnull().any():
            if strategy == 'mean':
                fill_value = processed_data[col].mean()
            elif strategy == 'median':
                fill_value = processed_data[col].median()
            elif strategy == 'mode':
                fill_value = processed_data[col].mode()[0] if not processed_data[col].mode().empty else 0
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            processed_data[col] = processed_data[col].fillna(fill_value)
    
    return processed_data

def validate_dataframe(data, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and data types.
    Returns validation results dictionary.
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(data, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append('Input is not a pandas DataFrame')
        return validation_results
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    if numeric_columns:
        non_numeric = [col for col in numeric_columns 
                      if col in data.columns and not pd.api.types.is_numeric_dtype(data[col])]
        if non_numeric:
            validation_results['warnings'].append(f'Non-numeric columns specified as numeric: {non_numeric}')
    
    if data.empty:
        validation_results['warnings'].append('DataFrame is empty')
    
    return validation_results
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df[column]),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75)
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing NaN values and converting to appropriate types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            cleaned_df = cleaned_df.dropna(subset=[col])
    
    return cleaned_df.reset_index(drop=True)