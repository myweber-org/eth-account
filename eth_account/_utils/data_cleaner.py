
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_column(dataframe, column, method='minmax'):
    """
    Normalize a column using specified method.
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'minmax':
        min_val = dataframe[column].min()
        max_val = dataframe[column].max()
        if max_val == min_val:
            return dataframe[column].apply(lambda x: 0.5)
        normalized = (dataframe[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = dataframe[column].mean()
        std_val = dataframe[column].std()
        if std_val == 0:
            return dataframe[column].apply(lambda x: 0)
        normalized = (dataframe[column] - mean_val) / std_val
    
    elif method == 'robust':
        median_val = dataframe[column].median()
        iqr_val = stats.iqr(dataframe[column])
        if iqr_val == 0:
            return dataframe[column].apply(lambda x: 0)
        normalized = (dataframe[column] - median_val) / iqr_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5, 
                  normalize_method='minmax', drop_na=True):
    """
    Comprehensive data cleaning pipeline.
    """
    df = dataframe.copy()
    
    if drop_na:
        df = df.dropna()
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col, outlier_threshold)
    
    normalized_df = df.copy()
    for col in numeric_columns:
        if col in normalized_df.columns:
            normalized_df[f"{col}_normalized"] = normalize_column(
                normalized_df, col, normalize_method
            )
    
    return {
        'cleaned_data': df,
        'normalized_data': normalized_df,
        'original_shape': dataframe.shape,
        'cleaned_shape': df.shape,
        'rows_removed': dataframe.shape[0] - df.shape[0]
    }

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(dataframe) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True