
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
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def zscore_normalize(data, column):
    """
    Normalize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column]
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def minmax_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column]
    
    a, b = feature_range
    normalized = a + ((data[column] - min_val) * (b - a)) / (max_val - min_val)
    return normalized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_clean = data.copy()
    
    for col in columns:
        if data_clean[col].isnull().any():
            if strategy == 'mean':
                fill_value = data_clean[col].mean()
            elif strategy == 'median':
                fill_value = data_clean[col].median()
            elif strategy == 'mode':
                fill_value = data_clean[col].mode()[0]
            elif strategy == 'drop':
                data_clean = data_clean.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_clean[col] = data_clean[col].fillna(fill_value)
    
    return data_clean

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'feature1': np.random.normal(50, 15, n_samples),
        'feature2': np.random.exponential(10, n_samples),
        'feature3': np.random.uniform(0, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=10, replace=False)
    data.loc[missing_indices, 'feature1'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(n_samples, size=5, replace=False)
    data.loc[outlier_indices, 'feature2'] = data['feature2'].max() * 3
    
    return data

if __name__ == "__main__":
    # Example usage
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    print("\nMissing values per column:")
    print(sample_data.isnull().sum())
    
    # Handle missing values
    cleaned_data = handle_missing_values(sample_data, strategy='mean')
    print("\nAfter handling missing values:", cleaned_data.shape)
    
    # Remove outliers
    filtered_data, outliers_count = remove_outliers_iqr(cleaned_data, 'feature2')
    print(f"\nRemoved {outliers_count} outliers from feature2")
    print("Filtered data shape:", filtered_data.shape)
    
    # Normalize data
    filtered_data['feature1_normalized'] = zscore_normalize(filtered_data, 'feature1')
    filtered_data['feature3_normalized'] = minmax_normalize(filtered_data, 'feature3')
    
    print("\nData cleaning completed successfully")
    print("Final data shape:", filtered_data.shape)
    print("\nFirst 5 rows of cleaned data:")
    print(filtered_data.head())