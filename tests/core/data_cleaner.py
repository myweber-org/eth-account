
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5):
    """
    Comprehensive data cleaning function
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    stats_report = {}
    
    for col in numeric_columns:
        if col in df.columns:
            # Remove outliers
            cleaned_df, removed = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            
            # Normalize
            cleaned_df[f"{col}_normalized"] = normalize_minmax(cleaned_df, col)
            
            # Standardize
            cleaned_df[f"{col}_standardized"] = standardize_zscore(cleaned_df, col)
            
            stats_report[col] = {
                'original_rows': len(df),
                'cleaned_rows': len(cleaned_df),
                'outliers_removed': removed,
                'normalized_range': (cleaned_df[f"{col}_normalized"].min(), 
                                    cleaned_df[f"{col}_normalized"].max()),
                'standardized_mean': cleaned_df[f"{col}_standardized"].mean(),
                'standardized_std': cleaned_df[f"{col}_standardized"].std()
            }
    
    return cleaned_df, stats_report

def validate_data(df, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if not allow_nan:
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Dataset contains {nan_count} NaN values")
    
    return True

def example_usage():
    """
    Example usage of the data cleaning functions
    """
    # Create sample data
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    # Add some outliers
    df.loc[1000] = {'feature_a': 500, 'feature_b': 1000, 'category': 'A'}
    df.loc[1001] = {'feature_a': -200, 'feature_b': 5, 'category': 'B'}
    
    # Clean the dataset
    cleaned_df, report = clean_dataset(df, ['feature_a', 'feature_b'])
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    
    for col, stats in report.items():
        print(f"\n{col}:")
        print(f"  Outliers removed: {stats['outliers_removed']}")
        print(f"  Normalized range: {stats['normalized_range']}")
    
    return cleaned_df, report

if __name__ == "__main__":
    cleaned_data, statistics = example_usage()