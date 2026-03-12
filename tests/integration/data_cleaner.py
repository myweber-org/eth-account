
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, standardize_columns=True):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to drop duplicate rows
    standardize_columns (bool): Whether to standardize column names
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if standardize_columns:
        cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ', '_')
        print("Standardized column names to lowercase with underscores")
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
    
    return validation_results

def sample_data(df, n_samples=5, random_state=42):
    """
    Generate a sample of the DataFrame for inspection.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    n_samples (int): Number of samples to return
    random_state (int): Random seed for reproducibility
    
    Returns:
    pd.DataFrame: Sampled DataFrame
    """
    if len(df) <= n_samples:
        return df
    
    return df.sample(n=n_samples, random_state=random_state)

if __name__ == "__main__":
    sample_data = {
        'User ID': [1, 2, 2, 3, 4, 5],
        'First Name': ['John', 'Jane', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'Last Name': ['Doe', 'Smith', 'Smith', 'Johnson', 'Williams', 'Brown'],
        'Email': ['john@example.com', 'jane@example.com', 'jane@example.com', 
                  'bob@example.com', 'alice@example.com', 'charlie@example.com'],
        'Age': [25, 30, 30, 35, 28, 40]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    validation = validate_data(cleaned, required_columns=['user_id', 'email'])
    print("Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50 + "\n")
    print("Data Sample:")
    print(sample_data(cleaned))