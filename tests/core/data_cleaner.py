import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data.reset_index(drop=True)

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    
    return stats

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    return filtered_df

def normalize_minmax(dataframe, columns):
    """
    Normalize specified columns using min-max scaling
    """
    df_normalized = dataframe.copy()
    for col in columns:
        if col in dataframe.columns:
            min_val = dataframe[col].min()
            max_val = dataframe[col].max()
            if max_val != min_val:
                df_normalized[col] = (dataframe[col] - min_val) / (max_val - min_val)
    return df_normalized

def standardize_zscore(dataframe, columns):
    """
    Standardize specified columns using z-score normalization
    """
    df_standardized = dataframe.copy()
    for col in columns:
        if col in dataframe.columns:
            mean_val = dataframe[col].mean()
            std_val = dataframe[col].std()
            if std_val > 0:
                df_standardized[col] = (dataframe[col] - mean_val) / std_val
    return df_standardized

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy
    """
    df_processed = dataframe.copy()
    
    if columns is None:
        columns = dataframe.columns
    
    for col in columns:
        if col in dataframe.columns and dataframe[col].isnull().any():
            if strategy == 'mean':
                fill_value = dataframe[col].mean()
            elif strategy == 'median':
                fill_value = dataframe[col].median()
            elif strategy == 'mode':
                fill_value = dataframe[col].mode()[0]
            elif strategy == 'drop':
                df_processed = df_processed.dropna(subset=[col])
                continue
            else:
                fill_value = 0
            
            df_processed[col] = df_processed[col].fillna(fill_value)
    
    return df_processed

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content
    """
    if dataframe.empty:
        raise ValueError("DataFrame is empty")
    
    if len(dataframe) < min_rows:
        raise ValueError(f"DataFrame has fewer than {min_rows} rows")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True