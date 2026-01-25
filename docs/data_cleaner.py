
import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns=None, method='zscore'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
        method: normalization method ('zscore', 'minmax', 'robust')
    
    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'zscore':
            df_normalized[col] = stats.zscore(df[col])
        elif method == 'minmax':
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
        elif method == 'robust':
            col_median = df[col].median()
            col_iqr = stats.iqr(df[col])
            if col_iqr > 0:
                df_normalized[col] = (df[col] - col_median) / col_iqr
    
    return df_normalized

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to check for outliers
        method: outlier detection method ('iqr', 'zscore')
        threshold: threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    mask = pd.Series([True] * len(df))
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
            col_mask = z_scores < threshold
        
        mask = mask & col_mask
    
    return df_clean[mask].reset_index(drop=True)

def handle_missing_values(df, columns=None, strategy='mean'):
    """
    Handle missing values in specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to handle missing values
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_imputed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean':
            df_imputed[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median':
            df_imputed[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            df_imputed[col] = df[col].fillna(df[col].mode()[0])
        elif strategy == 'drop':
            df_imputed = df_imputed.dropna(subset=[col])
    
    return df_imputed

def clean_dataset(df, config=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        config: dictionary with cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    if config is None:
        config = {
            'missing_values': {'strategy': 'mean'},
            'outliers': {'method': 'iqr', 'threshold': 1.5},
            'normalization': {'method': 'zscore'}
        }
    
    df_clean = df.copy()
    
    # Handle missing values
    if 'missing_values' in config:
        df_clean = handle_missing_values(
            df_clean, 
            strategy=config['missing_values'].get('strategy', 'mean')
        )
    
    # Remove outliers
    if 'outliers' in config:
        df_clean = remove_outliers(
            df_clean,
            method=config['outliers'].get('method', 'iqr'),
            threshold=config['outliers'].get('threshold', 1.5)
        )
    
    # Normalize data
    if 'normalization' in config:
        df_clean = normalize_data(
            df_clean,
            method=config['normalization'].get('method', 'zscore')
        )
    
    return df_clean
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(df, numeric_columns):
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
        df = normalize_column(df, col)
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = 'raw_data.csv'
    output_file = 'cleaned_data.csv'
    numeric_cols = ['age', 'income', 'score']
    
    raw_df = load_dataset(input_file)
    cleaned_df = clean_data(raw_df, numeric_cols)
    save_cleaned_data(cleaned_df, output_file)
    print(f"Data cleaning completed. Saved to {output_file}")