
import pandas as pd
import numpy as np
from datetime import datetime

def clean_csv_data(input_path, output_path):
    """
    Clean CSV data by handling missing values and converting data types.
    """
    try:
        df = pd.read_csv(input_path)
        
        print(f"Original shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna('Unknown', inplace=True)
        
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        df.to_csv(output_path, index=False)
        
        print(f"Cleaned data saved to: {output_path}")
        print(f"Final shape: {df.shape}")
        print(f"Missing values after cleaning:\n{df.isnull().sum()}")
        
        return True
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

def validate_dataframe(df):
    """
    Validate dataframe for common data quality issues.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'duplicate_rows': df.duplicated().sum(),
        'missing_values': df.isnull().sum().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    for col in validation_results['numeric_columns']:
        validation_results[f'{col}_stats'] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return validation_results

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    success = clean_csv_data(input_file, output_file)
    
    if success:
        cleaned_df = pd.read_csv(output_file)
        validation = validate_dataframe(cleaned_df)
        
        print("\nData Validation Results:")
        for key, value in validation.items():
            print(f"{key}: {value}")