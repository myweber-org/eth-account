
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicates,
    standardizing column names, and handling missing values.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    # Fill missing numeric values with column median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Fill missing categorical values with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'unknown')
    
    return df_clean

def validate_dataframe(df):
    """
    Validate that DataFrame meets basic quality criteria.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if df.isnull().sum().sum() > 0:
        print("Warning: DataFrame contains missing values")
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Warning: DataFrame contains {duplicates} duplicate rows")
    
    return True

def process_data_file(input_path, output_path):
    """
    Read, clean, and save a data file.
    """
    try:
        # Read data
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.xlsx'):
            df = pd.read_excel(input_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Validate data
        validate_dataframe(df)
        
        # Clean data
        df_clean = clean_dataframe(df)
        
        # Save cleaned data
        df_clean.to_csv(output_path, index=False)
        print(f"Data cleaned and saved to {output_path}")
        
        return df_clean
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = process_data_file(input_file, output_file)
    
    if cleaned_df is not None:
        print(f"Data shape: {cleaned_df.shape}")
        print(f"Columns: {list(cleaned_df.columns)}")