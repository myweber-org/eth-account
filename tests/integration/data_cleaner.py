
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(filepath):
    df = pd.read_csv(filepath)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_minmax(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv')
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaned. Original shape: {pd.read_csv('raw_data.csv').shape}, Cleaned shape: {cleaned_data.shape}")
import numpy as np
import pandas as pd

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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def example_usage():
    """
    Example demonstrating how to use the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 50, 5)
        ])
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary statistics:")
    print(calculate_summary_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary statistics:")
    print(calculate_summary_stats(cleaned_df, 'values'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_dataimport numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using IQR method.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_data(data, column):
    """
    Standardize data using Z-score normalization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_method='zscore', normalize=False, standardize=False):
    """
    Main cleaning pipeline for numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        elif outlier_method == 'iqr':
            outliers = detect_outliers_iqr(cleaned_df, col)
            cleaned_df = cleaned_df.drop(outliers.index)
        
        if normalize:
            cleaned_df = normalize_minmax(cleaned_df, col)
        
        if standardize:
            cleaned_df = standardize_data(cleaned_df, col)
    
    return cleaned_df

def validate_cleaning(original_df, cleaned_df, numeric_columns):
    """
    Validate cleaning results by comparing statistics.
    """
    validation_report = {}
    
    for col in numeric_columns:
        if col in original_df.columns and col in cleaned_df.columns:
            original_stats = {
                'count': original_df[col].count(),
                'mean': original_df[col].mean(),
                'std': original_df[col].std(),
                'min': original_df[col].min(),
                'max': original_df[col].max()
            }
            
            cleaned_stats = {
                'count': cleaned_df[col].count(),
                'mean': cleaned_df[col].mean(),
                'std': cleaned_df[col].std(),
                'min': cleaned_df[col].min(),
                'max': cleaned_df[col].max()
            }
            
            validation_report[col] = {
                'original': original_stats,
                'cleaned': cleaned_stats,
                'rows_removed': original_stats['count'] - cleaned_stats['count']
            }
    
    return validation_report

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000)
    })
    
    cleaned_data = clean_dataset(
        sample_data, 
        ['feature_a', 'feature_b', 'feature_c'],
        outlier_method='zscore',
        normalize=True,
        standardize=False
    )
    
    report = validate_cleaning(sample_data, cleaned_data, ['feature_a', 'feature_b', 'feature_c'])
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(f"Rows removed: {sample_data.shape[0] - cleaned_data.shape[0]}")
    
    for col, stats in report.items():
        print(f"\n{col}:")
        print(f"  Rows removed: {stats['rows_removed']}")
        print(f"  Original std: {stats['original']['std']:.2f}")
        print(f"  Cleaned std: {stats['cleaned']['std']:.2f}")