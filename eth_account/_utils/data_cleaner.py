
import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = df.copy()
    
    cleaned_df = cleaned_df.dropna()
    
    cleaned_df = cleaned_df.drop_duplicates()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_clean_dataset(df):
    """
    Validate that the DataFrame has no null values or duplicates.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        bool: True if DataFrame is clean, False otherwise.
    """
    null_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    
    return null_count == 0 and duplicate_count == 0

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, None, 4, 1],
        'B': [5, 6, 7, None, 5],
        'C': [8, 9, 10, 11, 8]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nNull values:", df.isnull().sum().sum())
    print("Duplicates:", df.duplicated().sum())
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nValidation result:", validate_clean_dataset(cleaned_df))import pandas as pd
import numpy as np

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val - min_val == 0:
        return data[column]
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df

def main():
    sample_data = {'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 12, 10]}
    df = pd.DataFrame(sample_data)
    print("Original Data:")
    print(df)
    
    outliers = detect_outliers_iqr(df, 'values')
    print(f"\nDetected outliers:\n{outliers}")
    
    cleaned_df = clean_dataset(df, ['values'])
    print("\nCleaned Data:")
    print(cleaned_df)

if __name__ == "__main__":
    main()import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main cleaning function for numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, numeric_columns):
    """
    Validate data structure and content.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    for col in numeric_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column {col} must be numeric")
    
    return True

def get_summary_statistics(df, numeric_columns):
    """
    Generate summary statistics for numeric columns.
    """
    summary = {}
    for col in numeric_columns:
        if col in df.columns:
            summary[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': df[col].count(),
                'missing': df[col].isnull().sum()
            }
    return pd.DataFrame(summary).T

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    numeric_cols = ['feature1', 'feature2']
    
    try:
        validate_data(sample_data, numeric_cols, numeric_cols)
        cleaned_data = clean_dataset(sample_data, numeric_cols, 'iqr', 'zscore')
        summary_stats = get_summary_statistics(cleaned_data, numeric_cols)
        print("Data cleaning completed successfully")
        print(f"Original shape: {sample_data.shape}")
        print(f"Cleaned shape: {cleaned_data.shape}")
        print("\nSummary statistics:")
        print(summary_stats)
    except Exception as e:
        print(f"Error during data cleaning: {e}")