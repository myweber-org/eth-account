
import pandas as pd
import numpy as np

def clean_dataset(df, deduplicate=True, handle_nulls='drop', null_threshold=0.5):
    """
    Clean a pandas DataFrame by removing duplicates and handling null values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    deduplicate (bool): Whether to remove duplicate rows
    handle_nulls (str): Method to handle nulls - 'drop', 'fill', or 'threshold'
    null_threshold (float): Threshold for column null percentage when handle_nulls='threshold'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    # Remove duplicates if requested
    if deduplicate:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    # Handle null values based on specified method
    if handle_nulls == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with any null values")
    
    elif handle_nulls == 'fill':
        # Fill numeric columns with median, categorical with mode
        for column in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
            else:
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown')
        print("Filled null values with appropriate replacements")
    
    elif handle_nulls == 'threshold':
        # Drop columns with null percentage above threshold
        null_percentages = cleaned_df.isnull().sum() / len(cleaned_df)
        columns_to_drop = null_percentages[null_percentages > null_threshold].index
        cleaned_df = cleaned_df.drop(columns=columns_to_drop)
        print(f"Dropped {len(columns_to_drop)} columns with >{null_threshold*100}% null values")
        
        # Drop remaining rows with nulls
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with remaining null values")
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and nulls
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, deduplicate=True, handle_nulls='fill')
    print("\nCleaned DataFrame:")
    print(cleaned)