
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if data.empty:
        return {}
    
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    return stats

def process_dataset(data, column):
    """
    Complete pipeline for processing a dataset column.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    tuple: (cleaned_data, original_stats, cleaned_stats)
    """
    original_stats = calculate_summary_statistics(data, column)
    cleaned_data = remove_outliers_iqr(data, column)
    cleaned_stats = calculate_summary_statistics(cleaned_data, column)
    
    return cleaned_data, original_stats, cleaned_stats
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in dataframe.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): Specific columns to process, None for all numeric columns
    
    Returns:
    pd.DataFrame: Dataframe with missing values handled
    """
    cleaned_data = data.copy()
    
    if columns is None:
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in cleaned_data.columns:
            continue
            
        if strategy == 'drop':
            cleaned_data = cleaned_data.dropna(subset=[col])
        elif strategy == 'mean':
            cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mean())
        elif strategy == 'median':
            cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
        elif strategy == 'mode':
            cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return cleaned_data

def process_dataset(data, config):
    """
    Process dataset with multiple cleaning operations.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    config (dict): Configuration dictionary with processing steps
    
    Returns:
    pd.DataFrame: Processed dataframe
    """
    processed_data = data.copy()
    
    if 'remove_outliers' in config:
        for col in config['remove_outliers'].get('columns', []):
            multiplier = config['remove_outliers'].get('multiplier', 1.5)
            processed_data = remove_outliers_iqr(processed_data, col, multiplier)
    
    if 'normalize' in config:
        for col in config['normalize'].get('columns', []):
            method = config['normalize'].get('method', 'minmax')
            if method == 'minmax':
                processed_data[col] = normalize_minmax(processed_data, col)
            elif method == 'zscore':
                processed_data[col] = standardize_zscore(processed_data, col)
    
    if 'handle_missing' in config:
        strategy = config['handle_missing'].get('strategy', 'mean')
        columns = config['handle_missing'].get('columns', None)
        processed_data = clean_missing_values(processed_data, strategy, columns)
    
    return processed_data
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df, column_types):
    """
    Convert specified columns to given data types.
    
    Args:
        df: pandas DataFrame
        column_types: dict mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_converted = df.copy()
    for column, dtype in column_types.items():
        if column in df_converted.columns:
            df_converted[column] = df_converted[column].astype(dtype)
    return df_converted

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill with value
        fill_value: value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fill_value)
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def clean_dataframe(df, deduplicate=True, type_conversions=None, missing_strategy='drop'):
    """
    Comprehensive data cleaning function.
    
    Args:
        df: pandas DataFrame
        deduplicate: whether to remove duplicates
        type_conversions: dict of column type conversions
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, unique_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame
        required_columns: list of columns that must be present
        unique_columns: list of columns that should have unique values
    
    Returns:
        dict with validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'duplicate_values': {},
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    if unique_columns:
        for column in unique_columns:
            if column in df.columns:
                duplicates = df[column].duplicated().sum()
                if duplicates > 0:
                    validation_results['duplicate_values'][column] = duplicates
                    validation_results['is_valid'] = False
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None],
        'age': ['25', '30', '30', '35', '40'],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    cleaned = clean_dataframe(
        df,
        deduplicate=True,
        type_conversions={'age': 'int32', 'score': 'float32'},
        missing_strategy='fill'
    )
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(
        cleaned,
        required_columns=['id', 'name', 'age', 'score'],
        unique_columns=['id']
    )
    
    print("\nValidation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    remove_duplicates (bool): Whether to remove duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
        print("Filled missing values with mode")
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if len(df) < min_rows:
        print(f"Dataset has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.isnull().all().any():
        print("Some columns contain only missing values")
        return False
    
    return True

def get_dataset_summary(df):
    """
    Generate a summary of the dataset.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=['number']).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns)
    }
    
    return summary