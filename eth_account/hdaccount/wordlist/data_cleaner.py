
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def z_score_normalize(data, column):
    """
    Normalize data using Z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].copy()
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def min_max_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].copy()
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    
    if feature_range != (0, 1):
        new_min, new_max = feature_range
        normalized = normalized * (new_max - new_min) + new_min
    
    return normalized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for col in columns:
        if col not in data_copy.columns:
            continue
            
        if data_copy[col].isnull().any():
            if strategy == 'mean':
                fill_value = data_copy[col].mean()
            elif strategy == 'median':
                fill_value = data_copy[col].median()
            elif strategy == 'mode':
                fill_value = data_copy[col].mode()[0]
            elif strategy == 'drop':
                data_copy = data_copy.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_copy[col] = data_copy[col].fillna(fill_value)
    
    return data_copy

def validate_dataframe(data, required_columns=None):
    """
    Validate DataFrame structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.randint(1, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(df.index, 5), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 3), 'feature_b'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    
    cleaned_data = handle_missing_values(sample_data, strategy='mean')
    print("After handling missing values:", cleaned_data.shape)
    
    normalized_data = cleaned_data.copy()
    normalized_data['feature_a_norm'] = z_score_normalize(cleaned_data, 'feature_a')
    normalized_data['feature_b_norm'] = min_max_normalize(cleaned_data, 'feature_b')
    
    print("Data normalization completed")
    print("Normalized columns added:", ['feature_a_norm', 'feature_b_norm'])