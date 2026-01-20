import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing data in a DataFrame using specified strategy.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values. 
                       Options: 'mean', 'median', 'mode', 'drop', 'fill'
        columns (list): List of columns to apply cleaning. If None, applies to all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        return df
    
    df_cleaned = df.copy()
    
    if columns is None:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_cleaned.columns:
            continue
            
        if strategy == 'drop':
            df_cleaned = df_cleaned.dropna(subset=[col])
        elif strategy == 'mean':
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
        elif strategy == 'median':
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        elif strategy == 'mode':
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
        elif strategy == 'fill':
            df_cleaned[col] = df_cleaned[col].fillna(0)
    
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def load_and_clean_csv(filepath, **kwargs):
    """
    Load CSV file and clean missing data.
    
    Args:
        filepath (str): Path to CSV file
        **kwargs: Additional arguments passed to clean_missing_data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        is_valid, message = validate_dataframe(df)
        
        if not is_valid:
            raise ValueError(f"Data validation failed: {message}")
        
        return clean_missing_data(df, **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except Exception as e:
        raise RuntimeError(f"Error processing file: {str(e)}")

def calculate_statistics(df):
    """
    Calculate basic statistics for numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary containing statistics
    """
    if df.empty:
        return {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = {}
    
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'missing_count': df[col].isna().sum()
        }
    
    return stats

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (mean imputation):")
    cleaned_df = clean_missing_data(df, strategy='mean')
    print(cleaned_df)
    print("\nStatistics:")
    stats = calculate_statistics(cleaned_df)
    for col, col_stats in stats.items():
        print(f"{col}: {col_stats}")