
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
import pandas as pd
import re

def clean_dataframe(df, column_name):
    """
    Clean a specific column in a DataFrame by removing duplicates,
    stripping whitespace, and converting to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].str.lower()
    df.drop_duplicates(subset=[column_name], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def remove_special_characters(df, column_name):
    """
    Remove special characters from a column, keeping only alphanumeric and spaces.
    """
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', '', x))
    return df

def normalize_column(df, column_name, mapping):
    """
    Normalize column values based on a provided mapping dictionary.
    """
    df[column_name] = df[column_name].replace(mapping)
    return df

if __name__ == "__main__":
    sample_data = {'Name': [' Alice ', 'bob', 'Alice', 'Charlie!!', '  david  ']}
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    df = clean_dataframe(df, 'Name')
    print("\nAfter cleaning:")
    print(df)

    df = remove_special_characters(df, 'Name')
    print("\nAfter removing special characters:")
    print(df)

    mapping = {'alice': 'Alice', 'bob': 'Bob', 'charlie': 'Charlie', 'david': 'David'}
    df = normalize_column(df, 'Name', mapping)
    print("\nAfter normalization:")
    print(df)import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, overwrites input file
        subset (list, optional): Columns to consider for identifying duplicates
        keep (str): Which duplicate to keep - 'first', 'last', or False to drop all duplicates
    """
    try:
        df = pd.read_csv(input_file)
        
        if subset:
            df_clean = df.drop_duplicates(subset=subset, keep=keep)
        else:
            df_clean = df.drop_duplicates(keep=keep)
        
        if output_file is None:
            output_file = input_file
        
        df_clean.to_csv(output_file, index=False)
        print(f"Removed {len(df) - len(df_clean)} duplicate rows")
        print(f"Cleaned data saved to: {output_file}")
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)