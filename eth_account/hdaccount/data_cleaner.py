import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, fill_strategy='mean'):
    """
    Clean a CSV file by handling missing values and removing duplicates.
    
    Parameters:
    input_path (str): Path to the input CSV file.
    output_path (str, optional): Path to save cleaned CSV. Defaults to None.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'drop').
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        print(f"Removed {duplicates_removed} duplicate rows.")
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        if fill_strategy == 'drop':
            df.dropna(inplace=True)
            print("Removed rows with missing values.")
        else:
            for col in numeric_cols:
                if df[col].isnull().any():
                    if fill_strategy == 'mean':
                        fill_value = df[col].mean()
                    elif fill_strategy == 'median':
                        fill_value = df[col].median()
                    elif fill_strategy == 'mode':
                        fill_value = df[col].mode()[0]
                    else:
                        fill_value = 0
                    
                    df[col].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in '{col}' with {fill_strategy}: {fill_value:.2f}")
            
            for col in categorical_cols:
                if df[col].isnull().any():
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_value, inplace=True)
                    print(f"Filled missing values in '{col}' with mode: '{mode_value}'")
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        print(f"Final data shape: {df.shape}")
        
        # Save cleaned data if output path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            print(f"Validation warning: Column '{col}' contains infinite values.")
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'age': [25, 30, None, 35, 40, 40],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0]
    }
    
    # Create sample CSV
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data('sample_data.csv', 'cleaned_data.csv', fill_strategy='mean')
    
    # Validate the cleaned data
    if cleaned_df is not None:
        validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
        print("\nCleaned data preview:")
        print(cleaned_df.head())