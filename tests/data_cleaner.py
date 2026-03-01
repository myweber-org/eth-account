
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, missing_strategy='mean'):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str, optional): Path to save cleaned data. If None, returns DataFrame
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
    
    Returns:
    pd.DataFrame or None: Cleaned DataFrame if output_path is None, else saves to file
    """
    
    # Validate input file exists
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Store original shape for logging
    original_shape = df.shape
    
    # Handle missing values based on strategy
    if missing_strategy == 'mean':
        # Fill numeric columns with column mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
    elif missing_strategy == 'median':
        # Fill numeric columns with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
    elif missing_strategy == 'zero':
        # Fill all missing values with 0
        df = df.fillna(0)
        
    elif missing_strategy == 'drop':
        # Drop rows with any missing values
        df = df.dropna()
        
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    # Log cleaning results
    cleaned_shape = df.shape
    rows_removed = original_shape[0] - cleaned_shape[0]
    print(f"Data cleaning complete:")
    print(f"  Original shape: {original_shape}")
    print(f"  Cleaned shape: {cleaned_shape}")
    print(f"  Rows removed: {rows_removed}")
    
    # Save or return results
    if output_path:
        output_file = Path(output_path)
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return None
    else:
        return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list, optional): List of required column names
    
    Returns:
    dict: Validation results with status and messages
    """
    
    validation_result = {
        'is_valid': True,
        'messages': [],
        'missing_columns': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['messages'].append('DataFrame is empty')
    
    # Check required columns if specified
    if required_columns:
        existing_columns = set(df.columns)
        required_set = set(required_columns)
        
        missing_columns = required_set - existing_columns
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = list(missing_columns)
            validation_result['messages'].append(
                f"Missing required columns: {list(missing_columns)}"
            )
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        validation_result['messages'].append(
            f"Found {duplicate_count} duplicate rows"
        )
    
    return validation_result

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, np.nan, 15.2, np.nan, 20.1],
        'category': ['A', 'B', 'A', np.nan, 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    print("\nCleaning with 'mean' strategy:")
    
    cleaned_df = clean_csv_data(
        input_path='dummy_path',  # Not actually used in this example
        missing_strategy='mean'
    )
    
    # For demonstration, apply cleaning directly
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    print(df)
    
    # Validate the cleaned data
    validation = validate_dataframe(df, required_columns=['id', 'value'])
    print(f"\nValidation result: {validation}")
import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing values in a DataFrame using specified strategy.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values. 
                       Options: 'mean', 'median', 'mode', 'drop', 'fill_zero'
        columns (list): List of columns to clean. If None, clean all columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].isnull().any():
            if strategy == 'mean':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
            elif strategy == 'fill_zero':
                df_clean[col].fillna(0, inplace=True)
    
    return df_clean

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
        method (str): Normalization method. Options: 'minmax', 'zscore'
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df_normalized[column].min()
        max_val = df_normalized[column].max()
        if max_val != min_val:
            df_normalized[column] = (df_normalized[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_normalized[column].mean()
        std_val = df_normalized[column].std()
        if std_val != 0:
            df_normalized[column] = (df_normalized[column] - mean_val) / std_val
    
    return df_normalized

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list): Columns to consider for duplicates
        keep (str): Which duplicates to keep. Options: 'first', 'last', False
    
    Returns:
        pd.DataFrame: DataFrame without duplicates
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        numeric_columns (list): List of columns that should be numeric
    
    Returns:
        dict: Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'non_numeric_columns': [],
        'null_counts': {},
        'shape': df.shape
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    if numeric_columns:
        non_numeric = [col for col in numeric_columns if col in df.columns 
                      and not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            validation_results['non_numeric_columns'] = non_numeric
            validation_results['is_valid'] = False
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_results['null_counts'][col] = null_count
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, np.nan, 50],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (mean imputation):")
    cleaned_df = clean_missing_data(df, strategy='mean')
    print(cleaned_df)
    
    validation = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
    print("\nValidation Results:")
    print(validation)