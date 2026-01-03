import pandas as pd

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing null values and duplicate rows.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    df_cleaned = df.copy()
    
    df_cleaned = df_cleaned.dropna()
    
    df_cleaned = df_cleaned.drop_duplicates()
    
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if validation passes.
    
    Raises:
        ValueError: If validation fails.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def process_data_file(file_path, required_columns=None):
    """
    Load and clean data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        required_columns (list): List of required column names.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        
        validate_dataframe(df, required_columns)
        
        df_clean = clean_dataframe(df)
        
        print(f"Original shape: {df.shape}")
        print(f"Cleaned shape: {df_clean.shape}")
        print(f"Rows removed: {len(df) - len(df_clean)}")
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'Bob', 'Charlie', None, 'Alice'],
        'age': [25, 30, 35, 40, 25],
        'city': ['NYC', 'LA', 'Chicago', 'Boston', 'NYC']
    }
    
    df_sample = pd.DataFrame(sample_data)
    print("Sample DataFrame:")
    print(df_sample)
    print("\nCleaned DataFrame:")
    df_cleaned = clean_dataframe(df_sample)
    print(df_cleaned)