
import pandas as pd
import numpy as np

def clean_dataset(df, drop_threshold=0.5, fill_strategy='median'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_threshold (float): Threshold for dropping columns with too many missing values
    fill_strategy (str): Strategy for filling missing values ('median', 'mean', 'mode')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Standardize column names
    cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Calculate missing percentage for each column
    missing_percent = cleaned_df.isnull().sum() / len(cleaned_df)
    
    # Drop columns with missing values above threshold
    columns_to_drop = missing_percent[missing_percent > drop_threshold].index
    cleaned_df = cleaned_df.drop(columns=columns_to_drop)
    
    # Fill remaining missing values based on strategy
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            if fill_strategy == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
            elif fill_strategy == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
            elif fill_strategy == 'mode':
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
            else:
                # For non-numeric columns or unknown strategy, fill with most frequent value
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
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
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def sample_usage():
    """Demonstrate usage of the data cleaning functions."""
    # Create sample data with missing values
    sample_data = {
        'Customer ID': [1, 2, 3, 4, 5],
        'First Name': ['Alice', 'Bob', None, 'David', 'Eve'],
        'Last Name': ['Smith', 'Johnson', 'Williams', None, 'Brown'],
        'Age': [25, 30, None, 35, 40],
        'Salary': [50000, None, 70000, 80000, 90000],
        'Department': ['Sales', 'IT', 'IT', None, 'HR']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Clean the data
    cleaned_df = clean_dataset(df, drop_threshold=0.3, fill_strategy='median')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned_df)
    print(f"\nValidation: {message}")

if __name__ == "__main__":
    sample_usage()