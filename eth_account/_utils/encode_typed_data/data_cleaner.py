import pandas as pd
import re

def clean_dataframe(df, text_columns=None, drop_duplicates=True, lowercase=True, strip_whitespace=True):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        text_columns (list, optional): List of column names to apply text cleaning.
            If None, all object dtype columns are cleaned.
        drop_duplicates (bool): Whether to remove duplicate rows.
        lowercase (bool): Convert text to lowercase.
        strip_whitespace (bool): Remove leading/trailing whitespace.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if text_columns is None:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in text_columns:
        if col in cleaned_df.columns:
            if lowercase:
                cleaned_df[col] = cleaned_df[col].astype(str).str.lower()
            if strip_whitespace:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
    
    return cleaned_df

def remove_special_characters(text, keep_pattern=r'[^a-zA-Z0-9\s]'):
    """
    Remove special characters from text.
    
    Args:
        text (str): Input text.
        keep_pattern (str): Regex pattern of characters to keep.
    
    Returns:
        str: Cleaned text.
    """
    if pd.isna(text):
        return text
    return re.sub(keep_pattern, '', str(text))

def validate_email(email):
    """
    Validate email format using regex.
    
    Args:
        email (str): Email address to validate.
    
    Returns:
        bool: True if email format is valid.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if pd.isna(email):
        return False
    return bool(re.match(pattern, str(email)))
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

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

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values using specified strategy
    """
    if strategy not in ['mean', 'median', 'mode', 'drop']:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    if strategy == 'drop':
        return data.dropna()
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            
            data[col] = data[col].fillna(fill_value)
    
    return data

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'feature3': np.random.randint(1, 100, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(df.index, 5), 'feature1'] = np.nan
    df.loc[np.random.choice(df.index, 3), 'feature2'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    print("\nMissing values per column:")
    print(sample_data.isnull().sum())
    
    cleaned_data = handle_missing_values(sample_data, 'mean')
    print("\nAfter handling missing values:", cleaned_data.shape)
    
    normalized_feature = normalize_minmax(cleaned_data, 'feature1')
    print("\nNormalized feature1 (first 5 values):")
    print(normalized_feature.head())
    
    standardized_feature = standardize_zscore(cleaned_data, 'feature2')
    print("\nStandardized feature2 (first 5 values):")
    print(standardized_feature.head())