import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and dropping specified columns.
    
    Args:
        filepath (str): Path to the CSV file
        missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
        columns_to_drop (list): List of column names to remove from dataset
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(filepath)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')
        
        if missing_strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif missing_strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif missing_strategy == 'drop':
            df = df.dropna()
        
        df = df.reset_index(drop=True)
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pandas.DataFrame): Dataframe to validate
        required_columns (list): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "Dataframe is empty or None"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataframe is valid"

def export_cleaned_data(df, output_path, format='csv'):
    """
    Export cleaned dataframe to file.
    
    Args:
        df (pandas.DataFrame): Dataframe to export
        output_path (str): Path for output file
        format (str): Output format ('csv' or 'excel')
    
    Returns:
        bool: True if export successful, False otherwise
    """
    try:
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        else:
            print(f"Unsupported format: {format}")
            return False
        
        print(f"Data successfully exported to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error exporting data: {str(e)}")
        return False

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': ['a', 'b', 'c', 'd', 'e'],
        'D': [10, 20, 30, 40, 50]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', missing_strategy='mean', columns_to_drop=['D'])
    
    if cleaned_df is not None:
        is_valid, message = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
        print(f"Validation: {is_valid} - {message}")
        
        if is_valid:
            export_success = export_cleaned_data(cleaned_df, 'cleaned_data.csv')
            print(f"Export successful: {export_success}")
    
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')