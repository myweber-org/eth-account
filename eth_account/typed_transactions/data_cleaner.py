import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicate rows and
    filling missing values with column means for numeric columns.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates()
    
    # Fill missing values for numeric columns with column mean
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    
    # For non-numeric columns, fill with mode (most frequent value)
    non_numeric_cols = df_clean.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        if df_clean[col].isnull().any():
            mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else ''
            df_clean[col] = df_clean[col].fillna(mode_value)
    
    return df_clean

def validate_data(df, required_columns=None):
    """
    Validate that the DataFrame meets basic requirements.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = pd.DataFrame({
#         'A': [1, 2, 2, None, 5],
#         'B': [None, 2, 3, 4, 5],
#         'C': ['a', 'b', 'b', None, 'e']
#     })
#     
#     cleaned = clean_dataset(sample_data)
#     print("Original data:")
#     print(sample_data)
#     print("\nCleaned data:")
#     print(cleaned)