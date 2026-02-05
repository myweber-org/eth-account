import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicate rows and
    filling missing values with appropriate defaults.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing values
    # For numeric columns, fill with median
    # For categorical columns, fill with mode
    for column in df_cleaned.columns:
        if pd.api.types.is_numeric_dtype(df_cleaned[column]):
            median_value = df_cleaned[column].median()
            df_cleaned[column] = df_cleaned[column].fillna(median_value)
        else:
            mode_value = df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else 'Unknown'
            df_cleaned[column] = df_cleaned[column].fillna(mode_value)
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_data(df, required_columns=None):
    """
    Validate that the DataFrame meets basic quality requirements.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = pd.DataFrame({
#         'A': [1, 2, None, 4, 1],
#         'B': ['x', 'y', None, 'x', 'z'],
#         'C': [10.5, 20.3, 15.7, None, 10.5]
#     })
#     
#     cleaned_data = clean_dataset(sample_data)
#     print("Original data:")
#     print(sample_data)
#     print("\nCleaned data:")
#     print(cleaned_data)