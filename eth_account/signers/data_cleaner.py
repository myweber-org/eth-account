
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'zero')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing and cleaned_df.isnull().sum().any():
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"Filled missing values in '{column}' with {strategy}: {fill_value}")
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if len(df) < min_rows:
        print(f"Dataset has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', 'C', 'A']
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    print("\nDataset info:")
    print(df.info())
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean')
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    # Validate the cleaned dataset
    is_valid = validate_dataset(cleaned_df, required_columns=['id', 'value'], min_rows=3)
    print(f"\nDataset validation: {'PASS' if is_valid else 'FAIL'}")

if __name__ == "__main__":
    main()