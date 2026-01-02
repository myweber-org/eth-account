import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def fill_missing_with_median(df, column):
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)
    return df

def standardize_column(df, column):
    mean = df[column].mean()
    std = df[column].std()
    df[column] = (df[column] - mean) / std
    return df

def clean_dataset(df, numeric_columns):
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
        df = fill_missing_with_median(df, col)
        df = standardize_column(df, col)
    return dfimport pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name):
    """
    Clean a numeric column by removing non-numeric characters.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = pd.to_numeric(df[column_name].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David'],
        'value': ['$100', '200', '200', '300USD', '400.50']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = remove_duplicates(df, subset=['id', 'name'])
    print("\nAfter removing duplicates:")
    print(cleaned_df)
    
    cleaned_df = clean_numeric_column(cleaned_df, 'value')
    print("\nAfter cleaning numeric column:")
    print(cleaned_df)
    
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'value'])
    print(f"\nValidation result: {is_valid}, Message: {message}")

if __name__ == "__main__":
    main()import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

def process_data(file_path, output_path=None):
    """
    Load, clean, and optionally save processed data.
    
    Args:
        file_path (str): Path to input data file.
        output_path (str): Path to save cleaned data.
    
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        
        is_valid, message = validate_data(df)
        if not is_valid:
            raise ValueError(f"Data validation failed: {message}")
        
        cleaned_df = clean_dataset(df)
        
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return cleaned_df
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The file is empty")
        raise
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned = clean_dataset(sample_data)
    print("\nCleaned data:")
    print(cleaned)