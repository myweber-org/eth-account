
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        columns_to_check (list, optional): Specific columns to check for duplicates. 
                                          If None, checks all columns.
        fill_missing (str): Method to fill missing values - 'mean', 'median', or 'drop'
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Remove duplicate rows
    if columns_to_check:
        df_cleaned = df.drop_duplicates(subset=columns_to_check)
    else:
        df_cleaned = df.drop_duplicates()
    
    # Handle missing values
    if fill_missing == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if fill_missing == 'mean':
                fill_value = df_cleaned[col].mean()
            else:  # median
                fill_value = df_cleaned[col].median()
            
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    
    # Report cleaning statistics
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Missing values handled using: {fill_missing}")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: Less than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing columns {missing_cols}")
            return False
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
#         'category': ['A', 'B', 'B', 'C', 'A', 'B']
#     }
#     
#     df = pd.DataFrame(data)
#     cleaned_df = clean_dataset(df, columns_to_check=['id'], fill_missing='mean')
#     
#     if validate_dataframe(cleaned_df, required_columns=['id', 'value']):
#         print("Data validation passed")
#         print(cleaned_df)