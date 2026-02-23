
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        dataframe: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        Cleaned DataFrame with duplicates removed
    """
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    
    return dataframe

def validate_dataframe(dataframe, required_columns):
    """
    Validate that DataFrame contains required columns.
    
    Args:
        dataframe: pandas DataFrame
        required_columns: list of required column names
    
    Returns:
        Boolean indicating if validation passed
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    return True
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
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
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of removal statistics for each column
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    removal_stats = {}
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            removal_stats[column] = {
                'removed': removed_count,
                'remaining': len(cleaned_df),
                'removal_percentage': (removed_count / original_count) * 100
            }
    
    return cleaned_df, removal_statsimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas Series using the IQR method.
    Returns cleaned Series and outlier indices.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
    cleaned_data = data[mask].copy()
    outlier_indices = data[~mask].index.tolist()
    
    return cleaned_data, outlier_indices

def normalize_minmax(data, column):
    """
    Apply min-max normalization to a column.
    Returns Series with normalized values.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Apply z-score standardization to a column.
    Returns Series with standardized values.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, method='zscore', remove_outliers=True):
    """
    Main cleaning function for datasets.
    Supports outlier removal and normalization/standardization.
    """
    cleaned_df = df.copy()
    
    if remove_outliers:
        outlier_report = {}
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df, outliers = remove_outliers_iqr(cleaned_df, col)
                outlier_report[col] = len(outliers)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if method == 'minmax':
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
            elif method == 'zscore':
                cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, numeric_columns):
    """
    Validate dataset structure and content.
    Returns validation results dictionary.
    """
    validation_result = {
        'missing_columns': [],
        'non_numeric_columns': [],
        'missing_values': {},
        'is_valid': True
    }
    
    for col in required_columns:
        if col not in df.columns:
            validation_result['missing_columns'].append(col)
            validation_result['is_valid'] = False
    
    for col in numeric_columns:
        if col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                validation_result['non_numeric_columns'].append(col)
                validation_result['is_valid'] = False
            
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                validation_result['missing_values'][col] = missing_count
    
    return validation_result

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(2, 1000),
        'feature_c': np.random.randint(1, 100, 1000)
    })
    
    validation = validate_data(
        sample_data,
        required_columns=['feature_a', 'feature_b', 'feature_c'],
        numeric_columns=['feature_a', 'feature_b', 'feature_c']
    )
    
    print(f"Validation result: {validation['is_valid']}")
    
    cleaned = clean_dataset(
        sample_data,
        numeric_columns=['feature_a', 'feature_b', 'feature_c'],
        method='zscore',
        remove_outliers=True
    )
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")