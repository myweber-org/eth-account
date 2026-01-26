
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df: pandas DataFrame to clean
        columns_to_check: list of columns to check for duplicates (default: all columns)
        drop_duplicates: boolean indicating whether to drop duplicate rows
        fill_missing: strategy to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    original_shape = df.shape
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Handle duplicates
    if drop_duplicates:
        if columns_to_check:
            cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
        else:
            cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_missing != 'drop':
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    else:
        cleaned_df = cleaned_df.dropna()
    
    # Log cleaning results
    print(f"Original dataset shape: {original_shape}")
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Rows removed: {original_shape[0] - cleaned_df.shape[0]}")
    print(f"Columns: {original_shape[1]}")
    
    return cleaned_df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate the cleaned dataset for basic data quality checks.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of columns that must be present
        numeric_columns: list of columns that should be numeric
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'has_duplicates': df.duplicated().any(),
        'has_missing_values': df.isnull().any().any(),
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
    
    # Check numeric columns
    if numeric_columns:
        non_numeric_cols = []
        for col in numeric_columns:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                non_numeric_cols.append(col)
        validation_results['non_numeric_columns'] = non_numeric_cols
    
    return validation_results

# Example usage (commented out for production)
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10, 20, np.nan, 30, 40, 50],
        'category': ['A', 'B', 'C', 'C', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns_to_check=['id'], fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    validation = validate_data(cleaned_df, required_columns=['id', 'value'], numeric_columns=['id', 'value'])
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, col)
        
        if normalize_method == 'minmax':
            cleaned_data[col] = normalize_minmax(cleaned_data, col)
        elif normalize_method == 'zscore':
            cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate statistical summary of the dataset.
    """
    summary = {
        'rows': len(data),
        'columns': len(data.columns),
        'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(data.select_dtypes(include=['object']).columns),
        'missing_values': data.isnull().sum().to_dict(),
        'data_types': data.dtypes.to_dict()
    }
    
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['X', 'Y', 'Z'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("\nData summary:")
    summary = get_data_summary(sample_data)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'], outlier_method='iqr', normalize_method='minmax')
    print("\nCleaned data shape:", cleaned.shape)
    print("\nCleaned data summary:")
    cleaned_summary = get_data_summary(cleaned)
    for key, value in cleaned_summary.items():
        print(f"{key}: {value}")