import pandas as pd
import numpy as np
import re

def clean_csv_data(input_file, output_file):
    """
    Load a CSV file, perform cleaning operations, and save the cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Standardize column names: lowercase and replace spaces with underscores
        df.columns = [re.sub(r'\s+', '_', col.strip().lower()) for col in df.columns]
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill missing non-numeric values with 'unknown'
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            df[col] = df[col].fillna('unknown')
        
        # Remove leading/trailing whitespace from string columns
        for col in non_numeric_cols:
            df[col] = df[col].astype(str).str.strip()
        
        # Save cleaned data to new CSV file
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    cleaned_df = clean_csv_data(input_csv, output_csv)
    
    if cleaned_df is not None:
        print(f"Data cleaning completed. Shape: {cleaned_df.shape}")