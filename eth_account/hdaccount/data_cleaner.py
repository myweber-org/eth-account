
import pandas as pd
import numpy as np
import sys

def clean_csv(input_path, output_path, strategy='mean'):
    """
    Clean a CSV file by handling missing values.
    
    Parameters:
    input_path (str): Path to the input CSV file.
    output_path (str): Path to save the cleaned CSV file.
    strategy (str): Strategy for handling missing values.
                    Options: 'mean', 'median', 'mode', 'drop'.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Original data shape: {df.shape}")
        
        if strategy == 'drop':
            df_cleaned = df.dropna()
        elif strategy in ['mean', 'median']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if strategy == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            else:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            df_cleaned = df
        elif strategy == 'mode':
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
                else:
                    df[col] = df[col].fillna(df[col].mean())
            df_cleaned = df
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data shape: {df_cleaned.shape}")
        print(f"Cleaned data saved to: {output_path}")
        
        missing_report = df_cleaned.isnull().sum()
        if missing_report.sum() > 0:
            print("Warning: Some missing values remain:")
            print(missing_report[missing_report > 0])
        else:
            print("No missing values remaining.")
            
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python data_cleaner.py <input_file> <output_file> [strategy]")
        print("Strategies: mean, median, mode, drop (default: mean)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    strategy = sys.argv[3] if len(sys.argv) > 3 else 'mean'
    
    clean_csv(input_file, output_file, strategy)