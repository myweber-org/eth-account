
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