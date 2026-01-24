
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                cleaned_df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col].fillna('Unknown', inplace=True)
            print(f"Filled missing values in categorical column '{col}' with 'Unknown'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("DataFrame is empty.")
        return False
    
    print("DataFrame validation passed.")
    return True

def main():
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, 20.3, np.nan, 40.1, 50.0, 50.0, np.nan],
        'category': ['A', 'B', 'A', np.nan, 'C', 'C', 'B'],
        'score': [85, 92, 78, np.nan, 88, 88, 90]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    if validate_dataframe(df, required_columns=['id', 'value', 'category']):
        cleaned_df = clean_dataframe(df, drop_duplicates=True, fill_missing='median')
        print("\nCleaned DataFrame:")
        print(cleaned_df)
        
        print(f"\nOriginal shape: {df.shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'mean',
    columns_to_drop: Optional[list] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and dropping specified columns.
    
    Parameters:
    input_path: Path to input CSV file
    output_path: Path to save cleaned CSV file
    missing_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns_to_drop: List of column names to remove
    
    Returns:
    Cleaned DataFrame
    """
    
    df = pd.read_csv(input_path)
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if missing_strategy == 'mean':
                fill_value = df[col].mean()
            elif missing_strategy == 'median':
                fill_value = df[col].median()
            elif missing_strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif missing_strategy == 'drop':
                df = df.dropna(subset=[col])
                continue
            else:
                fill_value = 0
            
            df[col] = df[col].fillna(fill_value)
    
    df.to_csv(output_path, index=False)
    return df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame structure and content.
    
    Returns:
    Dictionary with validation results
    """
    validation_result = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_result

if __name__ == "__main__":
    sample_df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, 7, 8],
        'C': ['x', 'y', 'z', 'x']
    })
    
    sample_df.to_csv('sample_data.csv', index=False)
    
    cleaned = clean_csv_data(
        'sample_data.csv',
        'cleaned_data.csv',
        missing_strategy='mean',
        columns_to_drop=['C']
    )
    
    print("Validation results:", validate_dataframe(cleaned))