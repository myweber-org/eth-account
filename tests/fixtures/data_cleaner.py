
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        columns_to_check (list, optional): Specific columns to check for duplicates. 
                                          If None, checks all columns.
        fill_missing (bool): Whether to fill missing values with column mean (numeric)
                            or mode (categorical)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                # Fill numeric columns with mean
                mean_value = cleaned_df[column].mean()
                cleaned_df[column] = cleaned_df[column].fillna(mean_value)
            else:
                # Fill categorical columns with mode
                mode_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown'
                cleaned_df[column] = cleaned_df[column].fillna(mode_value)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of columns that must be present
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    # Clean the data
    cleaned = clean_dataset(df, columns_to_check=['id', 'name'])
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nShape after cleaning:", cleaned.shape)
    
    # Validate
    is_valid, message = validate_dataframe(cleaned, required_columns=['id', 'name', 'age'])
    print(f"\nValidation: {message}")