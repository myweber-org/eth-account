
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing=True, remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list): List of columns to check for duplicates, None checks all columns
    fill_missing (bool): Whether to fill missing values with column mean (numeric) or mode (categorical)
    remove_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    # Handle missing values
    if fill_missing:
        for column in df_clean.columns:
            if df_clean[column].dtype in [np.float64, np.int64]:
                # Fill numeric columns with mean
                df_clean[column].fillna(df_clean[column].mean(), inplace=True)
            else:
                # Fill categorical columns with mode
                if not df_clean[column].mode().empty:
                    df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
    
    # Remove duplicates
    if remove_duplicates:
        if columns_to_check:
            df_clean.drop_duplicates(subset=columns_to_check, inplace=True)
        else:
            df_clean.drop_duplicates(inplace=True)
    
    # Reset index after cleaning
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate that DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, None],
#         'value': [10.5, 20.3, 20.3, None, 40.1, 50.0],
#         'category': ['A', 'B', 'B', 'C', None, 'A']
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\n" + "="*50 + "\n")
#     
#     # Clean the data
#     cleaned_df = clean_dataset(df, columns_to_check=['id', 'value'])
#     print("Cleaned DataFrame:")
#     print(cleaned_df)
#     
#     # Validate the cleaned data
#     is_valid, message = validate_data(cleaned_df, required_columns=['id', 'value', 'category'])
#     print(f"\nValidation: {is_valid} - {message}")