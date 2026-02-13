
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not process column '{col}': {e}")
    
    return cleaned_df

def get_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name
    
    Returns:
        dict: Dictionary containing statistics
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

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data shape:", df.shape)
    print("Column A statistics:", get_statistics(df, 'A'))
    
    cleaned_df = clean_numeric_data(df, ['A', 'B'])
    print("Cleaned data shape:", cleaned_df.shape)
    print("Column A statistics after cleaning:", get_statistics(cleaned_df, 'A'))
import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from specified columns or entire DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all columns
    
    Returns:
        Cleaned DataFrame
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values using specified strategy.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: list of column names or None for all columns
    
    Returns:
        DataFrame with filled values
    """
    if columns is None:
        columns = df.columns
    
    df_filled = df.copy()
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_filled[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'constant':
                df_filled[col] = df[col].fillna(0)
        else:
            df_filled[col] = df[col].fillna(df[col].mode()[0])
    
    return df_filled

def detect_outliers_iqr(df, columns=None):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        Dictionary with outlier indices per column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outliers = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outliers[col] = list(outlier_indices)
    
    return outliers

def remove_outliers(df, columns=None):
    """
    Remove rows containing outliers in specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        DataFrame with outliers removed
    """
    outliers = detect_outliers_iqr(df, columns)
    
    all_outlier_indices = set()
    for indices in outliers.values():
        all_outlier_indices.update(indices)
    
    return df.drop(index=list(all_outlier_indices))

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize data using specified method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
        method: 'minmax' or 'zscore'
    
    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def clean_dataset(df, missing_strategy='mean', remove_outliers_flag=True, normalize_flag=False):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        missing_strategy: strategy for handling missing values
        remove_outliers_flag: whether to remove outliers
        normalize_flag: whether to normalize data
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    cleaned_df = fill_missing_values(cleaned_df, strategy=missing_strategy)
    
    if remove_outliers_flag:
        cleaned_df = remove_outliers(cleaned_df)
    
    if normalize_flag:
        cleaned_df = normalize_data(cleaned_df)
    
    return cleaned_df