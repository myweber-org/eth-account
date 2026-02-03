
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Strategy to fill missing values. 
                       Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    missing_before = df.isnull().sum().sum()
    
    if fill_missing == 'drop':
        df = df.dropna()
        print(f"Dropped rows with missing values. Removed {missing_before} missing entries")
    elif fill_missing in ['mean', 'median', 'mode']:
        for column in df.select_dtypes(include=[np.number]).columns:
            if df[column].isnull().any():
                if fill_missing == 'mean':
                    fill_value = df[column].mean()
                elif fill_missing == 'median':
                    fill_value = df[column].median()
                elif fill_missing == 'mode':
                    fill_value = df[column].mode()[0] if not df[column].mode().empty else 0
                
                df[column] = df[column].fillna(fill_value)
                print(f"Filled missing values in column '{column}' with {fill_missing}: {fill_value}")
    
    missing_after = df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}, after: {missing_after}")
    print(f"Original shape: {original_shape}, Cleaned shape: {df.shape}")
    
    return df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    print("DataFrame validation passed")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'A', 'C', np.nan]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    if validate_dataframe(df, required_columns=['id', 'value']):
        cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
        print("\nCleaned DataFrame:")
        print(cleaned_df)