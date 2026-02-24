
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list): List of column names to check for duplicates.
                            If None, checks all columns.
    fill_missing (str): Method to fill missing values.
                       Options: 'mean', 'median', 'mode', 'drop', or a numeric value.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    original_shape = df.shape
    
    # Remove duplicates
    if columns_to_check is None:
        df_cleaned = df.drop_duplicates()
    else:
        df_cleaned = df.drop_duplicates(subset=columns_to_check)
    
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    
    # Handle missing values
    if fill_missing == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if fill_missing == 'mean':
                fill_value = df_cleaned[col].mean()
            elif fill_missing == 'median':
                fill_value = df_cleaned[col].median()
            elif fill_missing == 'mode':
                fill_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 0
            
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    elif isinstance(fill_missing, (int, float)):
        df_cleaned = df_cleaned.fillna(fill_missing)
    
    missing_filled = df.isna().sum().sum() - df_cleaned.isna().sum().sum()
    
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Missing values filled: {missing_filled}")
    
    return df_cleaned

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
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
            print(f"Validation failed: Missing columns: {missing_cols}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    data = {
        'id': [1, 2, 3, 1, 2, 6],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'David'],
        'age': [25, 30, np.nan, 25, 30, 35],
        'score': [85.5, 92.0, 78.5, 85.5, 92.0, 88.0]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'], fill_missing='mean')
    
    print("\n" + "="*50 + "\n")
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid = validate_data(cleaned_df, required_columns=['id', 'name', 'age', 'score'], min_rows=1)
    print(f"\nData validation passed: {is_valid}")