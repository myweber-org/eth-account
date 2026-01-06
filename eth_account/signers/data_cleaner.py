import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names (if None, auto-detect)
        outlier_factor: IQR multiplier for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            # Remove outliers
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_factor)
            
            # Normalize the column
            cleaned_data[f"{column}_normalized"] = normalize_minmax(cleaned_data, column)
            cleaned_data[f"{column}_standardized"] = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate data structure and content.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        allow_nan: whether to allow NaN values
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if not allow_nan and data.isnull().any().any():
        nan_cols = data.columns[data.isnull().any()].tolist()
        return False, f"NaN values found in columns: {nan_cols}"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    
    # Clean the data
    cleaned = clean_dataset(sample_data, numeric_columns=['feature1', 'feature2'])
    print("Cleaned data shape:", cleaned.shape)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, allow_nan=False)
    print(f"Validation: {is_valid}, Message: {message}")
import pandas as pd
import hashlib

def remove_duplicates_by_hash(df, columns):
    """
    Remove duplicate rows based on a hash of specified columns.
    """
    if not columns:
        raise ValueError("Columns list cannot be empty")
    
    df['_hash'] = df[columns].apply(
        lambda row: hashlib.md5(
            ''.join(row.astype(str)).encode()
        ).hexdigest(),
        axis=1
    )
    
    df_cleaned = df.drop_duplicates(subset=['_hash'])
    df_cleaned = df_cleaned.drop(columns=['_hash'])
    
    return df_cleaned.reset_index(drop=True)

def clean_numeric_columns(df, columns, fill_method='mean'):
    """
    Clean numeric columns by filling missing values.
    """
    for col in columns:
        if col not in df.columns:
            continue
            
        if fill_method == 'mean':
            fill_value = df[col].mean()
        elif fill_method == 'median':
            fill_value = df[col].median()
        elif fill_method == 'mode':
            fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
        else:
            fill_value = 0
        
        df[col] = df[col].fillna(fill_value)
    
    return df

def main():
    # Example usage
    data = {
        'id': [1, 2, 3, 4, 5, 3],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Charlie'],
        'age': [25, 30, None, 35, 40, 35],
        'score': [85.5, 92.0, 78.5, None, 88.0, 78.5]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Remove duplicates
    df_unique = remove_duplicates_by_hash(df, ['name', 'age', 'score'])
    print("After removing duplicates:")
    print(df_unique)
    print()
    
    # Clean numeric columns
    df_cleaned = clean_numeric_columns(df_unique, ['age', 'score'], fill_method='mean')
    print("After cleaning numeric columns:")
    print(df_cleaned)

if __name__ == "__main__":
    main()