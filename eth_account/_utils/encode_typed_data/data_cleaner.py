import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")
    
    if df.isnull().sum().any():
        missing_counts = df.isnull().sum()
        print(f"Missing values detected:\n{missing_counts[missing_counts > 0]}")
        
        if fill_missing == 'mean':
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_missing == 'median':
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_missing == 'mode':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
        elif fill_missing == 'drop':
            df = df.dropna()
            print("Rows with missing values have been dropped.")
        else:
            print(f"Warning: Unsupported fill_missing method '{fill_missing}'. Missing values retained.")
    
    print(f"Data cleaned. Original shape: {original_shape}, New shape: {df.shape}")
    return df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    if df.empty:
        raise ValueError("DataFrame is empty.")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows.")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 5],
        'B': [10, None, 10, 20, 30, 40],
        'C': ['x', 'y', 'x', 'z', None, 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_dataframe(cleaned_df, required_columns=['A', 'B'])
        print("Data validation passed.")
    except Exception as e:
        print(f"Data validation failed: {e}")
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to save cleaned CSV file
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        df_cleaned = df.copy()
        
        if missing_strategy == 'drop':
            df_cleaned = df_cleaned.dropna()
        elif missing_strategy in ['mean', 'median']:
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if df_cleaned[col].isnull().any():
                    if missing_strategy == 'mean':
                        fill_value = df_cleaned[col].mean()
                    else:
                        fill_value = df_cleaned[col].median()
                    
                    df_cleaned[col] = df_cleaned[col].fillna(fill_value)
        
        df_cleaned = df_cleaned.drop_duplicates()
        
        print(f"Cleaned data shape: {df_cleaned.shape}")
        print(f"Missing values after cleaning:\n{df_cleaned.isnull().sum()}")
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    if df.isnull().any().any():
        print("Validation warning: DataFrame contains missing values")
    
    return True

if __name__ == "__main__":
    cleaned_df = clean_csv_data('input_data.csv', 'cleaned_data.csv', 'mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        print(f"Data validation result: {'Passed' if is_valid else 'Failed'}")