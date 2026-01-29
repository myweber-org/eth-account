
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    file_path (str): Path to input CSV file.
    output_path (str): Path for cleaned CSV output. If None, returns DataFrame.
    missing_strategy (str): Strategy for handling missing values: 
                           'mean', 'median', 'mode', or 'drop'.
    
    Returns:
    pd.DataFrame or None: Cleaned DataFrame if output_path is None.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        original_rows = df.shape[0]
        df = df.drop_duplicates()
        removed_duplicates = original_rows - df.shape[0]
        print(f"Removed {removed_duplicates} duplicate rows")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'drop':
            df = df.dropna()
            print("Removed rows with missing values")
        elif missing_strategy in ['mean', 'median', 'mode']:
            for col in numeric_cols:
                if df[col].isnull().any():
                    if missing_strategy == 'mean':
                        fill_value = df[col].mean()
                    elif missing_strategy == 'median':
                        fill_value = df[col].median()
                    else:
                        fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
                    
                    df[col].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in '{col}' with {missing_strategy}: {fill_value:.2f}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            print(f"Final data shape: {df.shape}")
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    print(f"DataFrame validation passed. Shape: {df.shape}, Columns: {list(df.columns)}")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'value': [10.5, 20.3, np.nan, 15.7, 20.3, np.nan, 25.1, 30.4],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', missing_strategy='mean')
    
    if cleaned_df is not None:
        validation_result = validate_dataframe(cleaned_df, ['id', 'value', 'category'])
        print(f"Validation result: {validation_result}")
        
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')