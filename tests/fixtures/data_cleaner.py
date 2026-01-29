
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
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
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
        'count': df[column].count()
    }
    
    return stats

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
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 10 + 50
    }
    
    # Add some outliers
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[100] = [101, 200]  # Extreme outlier
    sample_df.loc[101] = [102, -100]  # Extreme outlier
    
    print("Original data shape:", sample_df.shape)
    print("Original statistics:", calculate_summary_statistics(sample_df, 'value'))
    
    cleaned_df = clean_numeric_data(sample_df, columns=['value'])
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'value'))import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using IQR method.
    Returns cleaned DataFrame and outlier indices.
    """
    df_clean = df.copy()
    outlier_indices = []
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices.extend(outliers)
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean, list(set(outlier_indices))

def normalize_data(df, columns, method='minmax'):
    """
    Normalize specified columns using different methods.
    """
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_normalized[col] = (df[col] - mean_val) / std_val
        
        elif method == 'robust':
            median_val = df[col].median()
            iqr_val = stats.iqr(df[col])
            if iqr_val > 0:
                df_normalized[col] = (df[col] - median_val) / iqr_val
    
    return df_normalized

def handle_missing_values(df, strategy='mean', fill_value=None):
    """
    Handle missing values in DataFrame.
    """
    df_processed = df.copy()
    
    for col in df.columns:
        if df[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                df_processed[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                df_processed[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_processed[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'constant' and fill_value is not None:
                df_processed[col] = df[col].fillna(fill_value)
            elif strategy == 'drop':
                df_processed = df_processed.dropna(subset=[col])
    
    return df_processed

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'non_numeric_columns': [],
        'empty_columns': []
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_result['missing_columns'] = missing
            validation_result['is_valid'] = False
    
    if numeric_columns:
        non_numeric = [col for col in numeric_columns if col in df.columns 
                      and not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            validation_result['non_numeric_columns'] = non_numeric
            validation_result['is_valid'] = False
    
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    if empty_cols:
        validation_result['empty_columns'] = empty_cols
        validation_result['is_valid'] = False
    
    return validation_result

def create_data_summary(df):
    """
    Create comprehensive data summary.
    """
    summary = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_stats'][col] = {
            'unique_count': df[col].nunique(),
            'top_value': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'top_count': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        }
    
    return summary