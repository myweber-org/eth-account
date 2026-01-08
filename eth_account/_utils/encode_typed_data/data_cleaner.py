
import pandas as pd
import numpy as np

def clean_dataset(df, text_columns=None, fill_na=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing text.
    
    Args:
        df: pandas DataFrame to clean
        text_columns: list of column names containing text data
        fill_na: boolean indicating whether to fill missing values
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    if fill_na:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        df_clean[categorical_cols] = df_clean[categorical_cols].fillna('Unknown')
    
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    df_clean = df_clean.drop_duplicates()
    
    return df_clean

def validate_data(df, required_columns=None):
    """
    Validate data integrity by checking for required columns and data types.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        tuple: (is_valid, validation_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        return False, f"Data contains null values: {null_counts[null_counts > 0].to_dict()}"
    
    return True, "Data validation passed"

def sample_data(df, sample_size=1000, random_state=42):
    """
    Create a random sample from the dataset for testing purposes.
    
    Args:
        df: pandas DataFrame to sample from
        sample_size: number of rows to sample
        random_state: random seed for reproducibility
    
    Returns:
        Sampled pandas DataFrame
    """
    if len(df) <= sample_size:
        return df
    
    return df.sample(n=sample_size, random_state=random_state)

if __name__ == "__main__":
    sample_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['John', 'Jane', None, 'Bob', 'Alice'],
        'age': [25, 30, 35, None, 28],
        'score': [85.5, 92.0, 78.5, 88.0, 95.5]
    })
    
    cleaned_df = clean_dataset(sample_df, text_columns=['name'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    is_valid, message = validate_data(cleaned_df, required_columns=['id', 'name', 'age'])
    print(f"\nValidation: {is_valid}")
    print(f"Message: {message}")
    
    sampled_df = sample_data(cleaned_df, sample_size=3)
    print(f"\nSampled DataFrame ({len(sampled_df)} rows):")
    print(sampled_df)
import pandas as pd
import numpy as np

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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def handle_missing_values(df, column, strategy='mean'):
    """
    Handle missing values in a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if strategy == 'mean':
        fill_value = df_copy[column].mean()
    elif strategy == 'median':
        fill_value = df_copy[column].median()
    elif strategy == 'mode':
        fill_value = df_copy[column].mode()[0]
    elif strategy == 'drop':
        df_copy = df_copy.dropna(subset=[column])
        return df_copy
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    df_copy[column] = df_copy[column].fillna(fill_value)
    
    return df_copy

def example_usage():
    """
    Example usage of the data cleaning utilities.
    """
    np.random.seed(42)
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 90),
            np.array([300, 350, -50, -100])
        ]),
        'category': np.random.choice(['A', 'B', 'C'], size=94)
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal statistics:")
    print(calculate_basic_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    print(calculate_basic_stats(cleaned_df, 'values'))
    
    normalized_df = normalize_column(cleaned_df, 'values', method='minmax')
    print("\nNormalized column sample:")
    print(normalized_df['values'].head())
    
    return cleaned_df

if __name__ == "__main__":
    result_df = example_usage()
    print("\nProcessing completed successfully.")