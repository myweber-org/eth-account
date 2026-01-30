
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    cleaned_df = dataframe.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if dataframe.empty:
        print("DataFrame is empty")
        return False
    
    return Trueimport pandas as pd
import numpy as np
import argparse
import os

def clean_csv(input_file, output_file=None, drop_na=False, fill_value=None):
    """
    Clean a CSV file by handling missing values and basic data validation.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {input_file} with {df.shape[0]} rows and {df.shape[1]} columns.")
        
        original_rows = df.shape[0]
        
        if drop_na:
            df = df.dropna()
            print(f"Removed rows with missing values. {df.shape[0]} rows remaining.")
        elif fill_value is not None:
            df = df.fillna(fill_value)
            print(f"Filled missing values with: {fill_value}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_cleaned.csv"
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        print(f"Original rows: {original_rows}, Cleaned rows: {df.shape[0]}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Clean CSV data files.')
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output CSV file path (optional)')
    parser.add_argument('--dropna', action='store_true', help='Drop rows with missing values')
    parser.add_argument('--fill', type=float, help='Fill missing numeric values with specified number')
    
    args = parser.parse_args()
    
    clean_csv(args.input, args.output, args.dropna, args.fill)

if __name__ == "__main__":
    main()