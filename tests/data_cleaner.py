
import pandas as pd
import numpy as np

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
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

def get_cleaning_stats(original_df, cleaned_df):
    """
    Get statistics about the cleaning process.
    
    Parameters:
    original_df (pd.DataFrame): Original DataFrame
    cleaned_df (pd.DataFrame): Cleaned DataFrame
    
    Returns:
    dict: Dictionary containing cleaning statistics
    """
    stats = {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'removed_rows': len(original_df) - len(cleaned_df),
        'removed_percentage': ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100
    }
    
    return stats

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame shape:", df.shape)
    
    cleaned_df = clean_numeric_data(df)
    
    print("Cleaned DataFrame shape:", cleaned_df.shape)
    
    stats = get_cleaning_stats(df, cleaned_df)
    print("Cleaning Statistics:", stats)import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, columns=None):
    """
    Normalize data using min-max scaling
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            col_min = data[col].min()
            col_max = data[col].max()
            
            if col_max != col_min:
                normalized_data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                normalized_data[col] = 0
    
    return normalized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    processed_data = data.copy()
    
    for col in columns:
        if col in data.columns and data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            processed_data[col] = data[col].fillna(fill_value)
    
    return processed_data

def clean_dataset(data, config=None):
    """
    Comprehensive data cleaning pipeline
    """
    if config is None:
        config = {
            'outlier_columns': None,
            'normalize_columns': None,
            'missing_strategy': 'mean'
        }
    
    cleaned_data = data.copy()
    
    if config.get('outlier_columns'):
        for col in config['outlier_columns']:
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    if config.get('missing_strategy'):
        cleaned_data = handle_missing_values(
            cleaned_data, 
            strategy=config['missing_strategy'],
            columns=config.get('missing_columns')
        )
    
    if config.get('normalize_columns'):
        cleaned_data = normalize_minmax(
            cleaned_data,
            columns=config['normalize_columns']
        )
    
    return cleaned_data

def validate_data(data, required_columns=None, min_rows=1):
    """
    Validate data structure and content
    """
    if len(data) < min_rows:
        raise ValueError(f"Data must have at least {min_rows} rows")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100) * 2 + 5,
        'C': np.random.randint(0, 100, 100)
    })
    
    sample_data.loc[10:15, 'A'] = np.nan
    
    config = {
        'outlier_columns': ['A', 'B'],
        'missing_strategy': 'median',
        'normalize_columns': ['A', 'B', 'C']
    }
    
    cleaned = clean_dataset(sample_data, config)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Missing values after cleaning: {cleaned.isnull().sum().sum()}")