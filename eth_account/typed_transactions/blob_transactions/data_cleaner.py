
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
    
    return filtered_df

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
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

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
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
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
        else:
            df_copy[f'{column}_normalized'] = 0.5
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
        else:
            df_copy[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 11, 10, 9, 8, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    print("\nSummary statistics:")
    print(calculate_summary_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned data (outliers removed):")
    print(cleaned_df)
    
    normalized_df = normalize_column(cleaned_df, 'values', method='minmax')
    print("\nNormalized data:")
    print(normalized_df)
import pandas as pd
import hashlib

def remove_duplicates(input_file, output_file, key_columns=None):
    """
    Remove duplicate rows from a CSV file based on specified columns.
    If key_columns is None, use all columns for comparison.
    """
    try:
        df = pd.read_csv(input_file)
        original_count = len(df)
        
        if key_columns is None:
            # Create a hash of all columns for comparison
            df['_hash'] = df.apply(lambda row: hashlib.md5(
                ''.join(str(row[col]) for col in df.columns).encode()
            ).hexdigest(), axis=1)
            df_clean = df.drop_duplicates(subset=['_hash'])
            df_clean = df_clean.drop(columns=['_hash'])
        else:
            # Check if specified columns exist
            missing_cols = [col for col in key_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}")
            df_clean = df.drop_duplicates(subset=key_columns)
        
        cleaned_count = len(df_clean)
        duplicates_removed = original_count - cleaned_count
        
        df_clean.to_csv(output_file, index=False)
        
        return {
            'original_rows': original_count,
            'cleaned_rows': cleaned_count,
            'duplicates_removed': duplicates_removed,
            'output_file': output_file
        }
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty.")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    # Check for NaN values in required columns
    if required_columns:
        nan_counts = df[required_columns].isna().sum()
        if nan_counts.any():
            return False, f"NaN values found in required columns: {dict(nan_counts[nan_counts > 0])}"
    
    return True, "DataFrame validation passed"

if __name__ == "__main__":
    # Example usage
    result = remove_duplicates(
        input_file='raw_data.csv',
        output_file='cleaned_data.csv',
        key_columns=['id', 'timestamp']
    )
    
    if result:
        print(f"Data cleaning completed:")
        print(f"  Original rows: {result['original_rows']}")
        print(f"  Cleaned rows: {result['cleaned_rows']}")
        print(f"  Duplicates removed: {result['duplicates_removed']}")
        print(f"  Output saved to: {result['output_file']}")
        
        # Validate the cleaned data
        df_clean = pd.read_csv(result['output_file'])
        is_valid, message = validate_dataframe(df_clean, ['id', 'value'])
        print(f"  Validation: {message}")