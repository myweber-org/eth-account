
import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns=None, method='zscore'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
        method: 'zscore', 'minmax', or 'robust'
    
    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'zscore':
            df_normalized[col] = (df[col] - df[col].mean()) / df[col].std()
        elif method == 'minmax':
            df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        elif method == 'robust':
            df_normalized[col] = (df[col] - df[col].median()) / stats.iqr(df[col])
    
    return df_normalized

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to check for outliers
        method: 'iqr' or 'zscore'
        threshold: multiplier for IQR or cutoff for z-score
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
            mask = z_scores < threshold
        
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of column names to process
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean':
            df_processed[col].fillna(df[col].mean(), inplace=True)
        elif strategy == 'median':
            df_processed[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'mode':
            df_processed[col].fillna(df[col].mode()[0], inplace=True)
        elif strategy == 'drop':
            df_processed = df_processed.dropna(subset=[col])
    
    return df_processed

def validate_data(df, checks=None):
    """
    Validate data quality with various checks.
    
    Args:
        df: pandas DataFrame
        checks: list of checks to perform
    
    Returns:
        Dictionary with validation results
    """
    if checks is None:
        checks = ['missing', 'duplicates', 'negative_values']
    
    results = {}
    
    if 'missing' in checks:
        results['missing_values'] = df.isnull().sum().to_dict()
    
    if 'duplicates' in checks:
        results['duplicate_rows'] = df.duplicated().sum()
    
    if 'negative_values' in checks:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        negative_counts = {}
        for col in numeric_cols:
            negative_counts[col] = (df[col] < 0).sum()
        results['negative_values'] = negative_counts
    
    return results
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values with different strategies
    """
    data_clean = data.copy()
    
    for column in data_clean.columns:
        if data_clean[column].isnull().any():
            if strategy == 'mean':
                fill_value = data_clean[column].mean()
            elif strategy == 'median':
                fill_value = data_clean[column].median()
            elif strategy == 'mode':
                fill_value = data_clean[column].mode()[0]
            elif strategy == 'drop':
                data_clean = data_clean.dropna(subset=[column])
                continue
            else:
                fill_value = 0
            
            data_clean[column] = data_clean[column].fillna(fill_value)
    
    return data_clean

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 200, 100)
    }
    
    # Add some outliers
    data['feature_a'][10] = 500
    data['feature_b'][20] = 800
    data['feature_c'][30] = -100
    
    # Add some missing values
    data['feature_a'][5] = np.nan
    data['feature_b'][15] = np.nan
    
    df = pd.DataFrame(data)
    return df

def main():
    """
    Demonstrate the data cleaning functions
    """
    print("Creating sample data...")
    df = create_sample_data()
    print(f"Original data shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    print("\nHandling missing values...")
    df_clean = handle_missing_values(df, strategy='mean')
    print(f"After cleaning shape: {df_clean.shape}")
    
    print("\nRemoving outliers using IQR...")
    df_iqr = remove_outliers_iqr(df_clean, 'feature_a')
    print(f"After IQR filtering shape: {df_iqr.shape}")
    
    print("\nNormalizing feature_a using Min-Max...")
    df_clean['feature_a_normalized'] = normalize_minmax(df_clean, 'feature_a')
    print(f"Normalized range: [{df_clean['feature_a_normalized'].min():.3f}, "
          f"{df_clean['feature_a_normalized'].max():.3f}]")
    
    print("\nNormalizing feature_b using Z-score...")
    df_clean['feature_b_standardized'] = normalize_zscore(df_clean, 'feature_b')
    print(f"Standardized mean: {df_clean['feature_b_standardized'].mean():.3f}, "
          f"std: {df_clean['feature_b_standardized'].std():.3f}")
    
    return df_clean

if __name__ == "__main__":
    cleaned_data = main()
    print(f"\nFinal cleaned data shape: {cleaned_data.shape}")
    print("Data cleaning completed successfully.")
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
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

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    tuple: (cleaned_df, removal_stats)
    """
    cleaned_df = df.copy()
    removal_stats = {}
    
    for column in columns_to_clean:
        if column in df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            removal_stats[column] = {
                'removed': removed_count,
                'remaining': len(cleaned_df),
                'percentage_removed': (removed_count / original_count) * 100
            }
    
    return cleaned_df, removal_stats

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'temperature': np.random.normal(25, 5, 1000),
        'humidity': np.random.normal(60, 15, 1000),
        'pressure': np.random.normal(1013, 10, 1000)
    })
    
    # Add some outliers
    sample_data.loc[50, 'temperature'] = 100
    sample_data.loc[150, 'humidity'] = 200
    sample_data.loc[250, 'pressure'] = 500
    
    print("Original dataset shape:", sample_data.shape)
    
    columns_to_clean = ['temperature', 'humidity', 'pressure']
    cleaned_data, stats = clean_dataset(sample_data, columns_to_clean)
    
    print("\nCleaned dataset shape:", cleaned_data.shape)
    print("\nOutlier removal statistics:")
    for col, stat in stats.items():
        print(f"{col}: {stat['removed']} outliers removed ({stat['percentage_removed']:.2f}%)")