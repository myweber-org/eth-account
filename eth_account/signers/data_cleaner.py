
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean', output_path=None):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Args:
        file_path (str): Path to input CSV file
        fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'drop')
        output_path (str, optional): Path to save cleaned CSV file
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            if fill_method == 'mean':
                df = df.fillna(df.mean(numeric_only=True))
            elif fill_method == 'median':
                df = df.fillna(df.median(numeric_only=True))
            elif fill_method == 'mode':
                df = df.fillna(df.mode().iloc[0])
            elif fill_method == 'drop':
                df = df.dropna()
            else:
                raise ValueError(f"Unknown fill method: {fill_method}")
            
            print(f"Missing values handled using '{fill_method}' method")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_numeric_columns(df, columns=None):
    """
    Validate that specified columns contain only numeric values.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        columns (list, optional): List of column names to validate
    
    Returns:
        dict: Validation results for each column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    results = {}
    for col in columns:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
            results[col] = {
                'total_values': len(df[col]),
                'non_numeric_count': non_numeric,
                'is_valid': non_numeric == 0
            }
    
    return results

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10.5, np.nan, 30.2, 40.1, 50.0],
        'C': ['x', 'y', 'z', 'x', 'y']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_method='mean', output_path='cleaned_data.csv')
    
    if cleaned_df is not None:
        validation = validate_numeric_columns(cleaned_df)
        print("Validation results:", validation)