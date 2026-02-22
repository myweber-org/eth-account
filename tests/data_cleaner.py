
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    data_clean = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if strategy == 'drop':
            data_clean = data_clean.dropna(subset=[col])
        elif strategy == 'mean':
            data_clean[col].fillna(data_clean[col].mean(), inplace=True)
        elif strategy == 'median':
            data_clean[col].fillna(data_clean[col].median(), inplace=True)
        elif strategy == 'mode':
            data_clean[col].fillna(data_clean[col].mode()[0], inplace=True)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return data_clean

def clean_dataset(data, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Args:
        data: pandas DataFrame
        config: dictionary with cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if 'remove_outliers' in config:
        for col, params in config['remove_outliers'].items():
            multiplier = params.get('multiplier', 1.5)
            cleaned_data = remove_outliers_iqr(cleaned_data, col, multiplier)
    
    if 'normalize' in config:
        for col in config['normalize']:
            if 'method' in config['normalize'][col]:
                method = config['normalize'][col]['method']
                if method == 'minmax':
                    cleaned_data[f"{col}_normalized"] = normalize_minmax(cleaned_data, col)
                elif method == 'zscore':
                    cleaned_data[f"{col}_standardized"] = standardize_zscore(cleaned_data, col)
    
    if 'handle_missing' in config:
        strategy = config['handle_missing'].get('strategy', 'mean')
        columns = config['handle_missing'].get('columns')
        cleaned_data = handle_missing_values(cleaned_data, strategy, columns)
    
    return cleaned_data

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 60],
        'C': [1.1, 2.2, np.nan, 4.4, 5.5, 6.6]
    })
    
    config = {
        'remove_outliers': {'A': {'multiplier': 1.5}},
        'normalize': {'B': {'method': 'minmax'}},
        'handle_missing': {'strategy': 'mean', 'columns': ['C']}
    }
    
    cleaned = clean_dataset(sample_data, config)
    print("Original data:")
    print(sample_data)
    print("\nCleaned data:")
    print(cleaned)