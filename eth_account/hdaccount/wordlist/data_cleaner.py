
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
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
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values with different strategies
    """
    if columns is None:
        columns = data.columns
    
    data_clean = data.copy()
    
    for column in columns:
        if column not in data.columns:
            continue
            
        if data[column].isnull().any():
            if strategy == 'mean':
                fill_value = data[column].mean()
            elif strategy == 'median':
                fill_value = data[column].median()
            elif strategy == 'mode':
                fill_value = data[column].mode()[0]
            elif strategy == 'drop':
                data_clean = data_clean.dropna(subset=[column])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
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
        'feature_c': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(df.index, size=5), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, size=3), 'feature_b'] = np.nan
    
    return df

def main():
    """
    Example usage of data cleaning functions
    """
    print("Creating sample data...")
    df = create_sample_data()
    print(f"Original data shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    print("\nHandling missing values...")
    df_clean = handle_missing_values(df, strategy='mean')
    print(f"After cleaning shape: {df_clean.shape}")
    
    print("\nRemoving outliers from feature_a...")
    df_filtered, removed = remove_outliers_iqr(df_clean, 'feature_a')
    print(f"Removed {removed} outliers")
    print(f"Filtered data shape: {df_filtered.shape}")
    
    print("\nNormalizing feature_b...")
    df_filtered['feature_b_normalized'] = normalize_minmax(df_filtered, 'feature_b')
    print("Normalization complete")
    
    print("\nStandardizing feature_a...")
    df_filtered['feature_a_standardized'] = standardize_zscore(df_filtered, 'feature_a')
    print("Standardization complete")
    
    print("\nSummary statistics:")
    print(df_filtered.describe())
    
    return df_filtered

if __name__ == "__main__":
    cleaned_data = main()