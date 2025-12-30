import pandas as pd
import argparse
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
        subset: Column names to consider for duplicates (optional)
        keep: Which duplicates to keep ('first', 'last', or False)
    
    Returns:
        Number of duplicates removed
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        final_count = len(df_cleaned)
        
        duplicates_removed = initial_count - final_count
        
        if output_file:
            df_cleaned.to_csv(output_file, index=False)
            print(f"Cleaned data saved to: {output_file}")
        else:
            df_cleaned.to_csv(input_file, index=False)
            print(f"Cleaned data saved to original file: {input_file}")
        
        print(f"Removed {duplicates_removed} duplicate rows")
        print(f"Original rows: {initial_count}, Cleaned rows: {final_count}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return -1
    except Exception as e:
        print(f"Error processing file: {e}")
        return -1

def main():
    parser = argparse.ArgumentParser(description='Remove duplicate rows from CSV files')
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output CSV file path (optional)')
    parser.add_argument('-s', '--subset', nargs='+', help='Column names to check for duplicates')
    parser.add_argument('-k', '--keep', choices=['first', 'last', 'none'], 
                       default='first', help='Which duplicates to keep')
    
    args = parser.parse_args()
    
    keep_value = 'first' if args.keep == 'first' else 'last' if args.keep == 'last' else False
    
    remove_duplicates(args.input, args.output, args.subset, keep_value)

if __name__ == "__main__":
    main()