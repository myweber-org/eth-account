
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