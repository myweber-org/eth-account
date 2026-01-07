
import pandas as pd
import re

def clean_text_column(df, column_name):
    """Standardize text by converting to lowercase and removing extra whitespace."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_email_column(df, column_name):
    """Validate email format in specified column."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['is_valid_email'] = df[column_name].apply(lambda x: bool(re.match(email_pattern, str(x))))
    return df

def process_dataframe(df, text_columns=None, email_column=None, deduplicate=True):
    """Main function to clean and process DataFrame."""
    if text_columns:
        for col in text_columns:
            df = clean_text_column(df, col)
    
    if email_column:
        df = validate_email_column(df, email_column)
    
    if deduplicate:
        df = remove_duplicates(df)
    
    return dfimport pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', drop_threshold=0.5):
    """
    Clean CSV data by handling missing values and removing columns with excessive nulls.
    
    Args:
        filepath (str): Path to the CSV file
        fill_method (str): Method for filling missing values ('mean', 'median', 'mode', or 'zero')
        drop_threshold (float): Threshold for dropping columns (0.0 to 1.0)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        
        # Calculate null percentages
        null_percentages = df.isnull().sum() / len(df)
        
        # Drop columns with null percentage above threshold
        columns_to_drop = null_percentages[null_percentages > drop_threshold].index
        df = df.drop(columns=columns_to_drop)
        
        # Fill remaining missing values
        for column in df.columns:
            if df[column].isnull().any():
                if fill_method == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                    df[column].fillna(df[column].mean(), inplace=True)
                elif fill_method == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                    df[column].fillna(df[column].median(), inplace=True)
                elif fill_method == 'mode':
                    df[column].fillna(df[column].mode()[0], inplace=True)
                elif fill_method == 'zero':
                    df[column].fillna(0, inplace=True)
                else:
                    # Forward fill as default for non-numeric columns
                    df[column].fillna(method='ffill', inplace=True)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    # Check for infinite values
    if np.any(np.isinf(df.select_dtypes(include=[np.number]))):
        print("Warning: DataFrame contains infinite values")
    
    return True

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame
        output_path (str): Path for output CSV file
    
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving cleaned data: {str(e)}")
        return False