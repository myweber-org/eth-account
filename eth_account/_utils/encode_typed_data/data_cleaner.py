
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values
                               ('mean', 'median', 'drop', 'zero')
    """
    try:
        df = pd.read_csv(input_file)
        
        original_rows = len(df)
        original_columns = len(df.columns)
        
        df = df.drop_duplicates()
        
        missing_counts = df.isnull().sum()
        
        if missing_strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mean(), inplace=True)
        
        elif missing_strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
        
        elif missing_strategy == 'zero':
            df.fillna(0, inplace=True)
        
        elif missing_strategy == 'drop':
            df.dropna(inplace=True)
        
        df = df.reset_index(drop=True)
        
        df.to_csv(output_file, index=False)
        
        cleaned_rows = len(df)
        duplicates_removed = original_rows - cleaned_rows
        
        print(f"Data cleaning completed successfully!")
        print(f"Original data: {original_rows} rows, {original_columns} columns")
        print(f"Cleaned data: {cleaned_rows} rows, {len(df.columns)} columns")
        print(f"Duplicates removed: {duplicates_removed}")
        print(f"Missing values handled using: {missing_strategy} strategy")
        print(f"Cleaned data saved to: {output_file}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Validation failed: Dataframe is None or empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Validation failed: Missing required columns: {missing_columns}")
            return False
    
    if df.isnull().any().any():
        print("Warning: Dataframe contains missing values")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', 'A', None, 'C', 'A']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_input.csv', index=False)
    
    cleaned_df = clean_csv_data('test_input.csv', 'test_output.csv', 'mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, ['id', 'value'])
        print(f"Data validation: {'PASSED' if is_valid else 'FAILED'}")