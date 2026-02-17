
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Load a CSV file, clean missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Original data shape: {df.shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Remove rows where all values are missing
        df = df.dropna(how='all')
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"cleaned_{input_file.name}"
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        print(f"Final data shape: {df.shape}")
        
        return df, output_path
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None, None

def validate_dataframe(df):
    """
    Perform basic validation on dataframe.
    """
    if df is None:
        return False
    
    validation_results = {
        'has_data': len(df) > 0,
        'has_columns': len(df.columns) > 0,
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum()
    }
    
    print("Data Validation Results:")
    for key, value in validation_results.items():
        print(f"  {key}: {value}")
    
    return all([validation_results['has_data'], validation_results['has_columns']])

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Alice'],
        'age': [25, 30, None, 40, 35, 25],
        'score': [85.5, 92.0, 78.5, None, 88.0, 85.5]
    }
    
    # Create temporary CSV for demonstration
    temp_df = pd.DataFrame(sample_data)
    temp_df.to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df, output_file = clean_csv_data('sample_data.csv')
    
    if cleaned_df is not None:
        validate_dataframe(cleaned_df)
    
    # Clean up temporary file
    import os
    if os.path.exists('sample_data.csv'):
        os.remove('sample_data.csv')