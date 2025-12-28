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

def clean_dataset_columns(df, columns_to_clean):
    """
    Clean specific columns in a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns_to_clean (list): List of column names to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned columns.
    """
    df_cleaned = df.copy()
    
    for column in columns_to_clean:
        if column in df_cleaned.columns:
            # Remove null values in specific column
            df_cleaned = df_cleaned[df_cleaned[column].notna()]
    
    return df_cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['Alice', 'Bob', 'Charlie', None, 'Alice'],
        'age': [25, 30, None, 40, 25],
        'city': ['NYC', 'LA', 'Chicago', 'Miami', 'NYC']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)