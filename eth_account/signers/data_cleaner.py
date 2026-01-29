import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        filepath: Path to the CSV file
        fill_method: Method for filling missing values ('mean', 'median', 'mode', 'zero')
        drop_threshold: Drop columns with missing values above this ratio (0.0 to 1.0)
    
    Returns:
        Cleaned DataFrame and cleaning report
    """
    try:
        df = pd.read_csv(filepath)
        original_shape = df.shape
        cleaning_report = {
            'original_rows': original_shape[0],
            'original_columns': original_shape[1],
            'missing_values': df.isnull().sum().sum(),
            'dropped_columns': []
        }
        
        # Drop columns with too many missing values
        missing_ratio = df.isnull().sum() / len(df)
        columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index.tolist()
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            cleaning_report['dropped_columns'] = columns_to_drop
        
        # Fill remaining missing values
        if fill_method == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif fill_method == 'median':
            df = df.fillna(df.median(numeric_only=True))
        elif fill_method == 'mode':
            df = df.fillna(df.mode().iloc[0])
        elif fill_method == 'zero':
            df = df.fillna(0)
        else:
            raise ValueError(f"Unsupported fill method: {fill_method}")
        
        cleaning_report['final_rows'] = df.shape[0]
        cleaning_report['final_columns'] = df.shape[1]
        cleaning_report['remaining_missing'] = df.isnull().sum().sum()
        
        return df, cleaning_report
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None, None

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required
    
    Returns:
        Boolean indicating if validation passed
    """
    if df is None or df.empty:
        return False
    
    if len(df) < min_rows:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def save_cleaned_data(df, output_path, index=False):
    """
    Save cleaned DataFrame to CSV.
    
    Args:
        df: Cleaned DataFrame
        output_path: Path to save the cleaned data
        index: Whether to include index in output
    
    Returns:
        Boolean indicating success
    """
    try:
        df.to_csv(output_path, index=index)
        print(f"Cleaned data saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df, report = clean_csv_data(input_file, fill_method='median', drop_threshold=0.3)
    
    if cleaned_df is not None:
        print(f"Cleaning Report: {report}")
        
        if validate_dataframe(cleaned_df, min_rows=10):
            save_cleaned_data(cleaned_df, output_file)
        else:
            print("Data validation failed")
    else:
        print("Data cleaning failed")