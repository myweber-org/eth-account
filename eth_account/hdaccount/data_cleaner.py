
import pandas as pd

def clean_dataset(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): If True, drop rows with any null values. Default True.
        column_case (str): Desired case for column names ('lower', 'upper', 'title'). Default 'lower'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Handle missing values
    if drop_na:
        df_clean = df_clean.dropna()
    else:
        df_clean = df_clean.fillna(0)
    
    # Standardize column names
    if column_case == 'lower':
        df_clean.columns = df_clean.columns.str.lower()
    elif column_case == 'upper':
        df_clean.columns = df_clean.columns.str.upper()
    elif column_case == 'title':
        df_clean.columns = df_clean.columns.str.title()
    
    # Remove any leading/trailing whitespace from column names
    df_clean.columns = df_clean.columns.str.strip()
    
    # Replace spaces with underscores in column names
    df_clean.columns = df_clean.columns.str.replace(' ', '_')
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
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

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'Name': ['Alice', 'Bob', None, 'David'],
#         'Age': [25, None, 30, 35],
#         'City': ['New York', 'London', 'Paris', 'Tokyo']
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned_df = clean_dataset(df, drop_na=True, column_case='lower')
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
#     
#     is_valid, message = validate_dataframe(cleaned_df, required_columns=['name', 'age'])
#     print(f"\nValidation: {is_valid}, Message: {message}")