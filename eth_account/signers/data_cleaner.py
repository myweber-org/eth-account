
import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for duplicates.
            If None, checks all columns. Defaults to None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Fill missing numeric values with column mean
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    # Fill missing categorical values with mode
    categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            mode_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown'
            cleaned_df[col] = cleaned_df[col].fillna(mode_value)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"