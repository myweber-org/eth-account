
import pandas as pd
import numpy as np
import re

def clean_csv_data(input_file, output_file):
    """
    Clean and preprocess CSV data by handling missing values,
    standardizing formats, and removing duplicates.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('unknown')
        
        # Clean string columns
        for col in categorical_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
        
        # Remove outliers using IQR method for numeric columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaning completed. Cleaned data saved to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        return False
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

def validate_data(file_path):
    """
    Validate the cleaned data file.
    """
    try:
        df = pd.read_csv(file_path)
        
        print("Data Validation Report:")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print("\nMissing values per column:")
        print(df.isnull().sum())
        print("\nData types:")
        print(df.dtypes)
        
        return True
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    if clean_csv_data(input_csv, output_csv):
        validate_data(output_csv)