
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
import pandas as pd
import hashlib

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: {'first', 'last', False} which duplicates to keep
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def generate_hash(row):
    """
    Generate MD5 hash for a row to identify duplicates.
    
    Args:
        row: pandas Series representing a row
    
    Returns:
        MD5 hash string
    """
    row_string = str(row.to_dict()).encode('utf-8')
    return hashlib.md5(row_string).hexdigest()

def clean_dataset(input_file, output_file):
    """
    Main function to clean dataset by removing duplicates.
    
    Args:
        input_file: path to input CSV file
        output_file: path to save cleaned CSV file
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original dataset shape: {df.shape}")
        
        df_cleaned = remove_duplicates(df)
        print(f"Cleaned dataset shape: {df_cleaned.shape}")
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Removed {len(df) - len(df_cleaned)} duplicate rows")
        print(f"Cleaned data saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")

if __name__ == "__main__":
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    clean_dataset(input_path, output_path)