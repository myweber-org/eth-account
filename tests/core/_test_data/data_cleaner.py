import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validate if the input is a pandas DataFrame and not empty.
    
    Args:
        df: Object to validate.
    
    Returns:
        bool: True if valid DataFrame, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty.")
        return False
    
    return True

def get_cleaning_report(df, df_cleaned):
    """
    Generate a report of the cleaning operations performed.
    
    Args:
        df (pd.DataFrame): Original DataFrame.
        df_cleaned (pd.DataFrame): Cleaned DataFrame.
    
    Returns:
        dict: Dictionary containing cleaning statistics.
    """
    original_rows = len(df)
    cleaned_rows = len(df_cleaned)
    null_rows = df.isnull().any(axis=1).sum()
    duplicate_rows = df.duplicated().sum()
    
    report = {
        'original_rows': original_rows,
        'cleaned_rows': cleaned_rows,
        'rows_removed': original_rows - cleaned_rows,
        'null_rows_removed': null_rows,
        'duplicate_rows_removed': duplicate_rows,
        'cleaning_percentage': ((original_rows - cleaned_rows) / original_rows * 100) if original_rows > 0 else 0
    }
    
    return report

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'A': [1, 2, None, 4, 2],
#         'B': [5, 6, 7, None, 6],
#         'C': [8, 9, 10, 11, 9]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     
#     if validate_dataframe(df):
#         cleaned_df = clean_dataset(df)
#         print("\nCleaned DataFrame:")
#         print(cleaned_df)
#         
#         report = get_cleaning_report(df, cleaned_df)
#         print("\nCleaning Report:")
#         for key, value in report.items():
#             print(f"{key}: {value}")