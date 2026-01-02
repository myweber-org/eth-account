
import pandas as pd
import re

def clean_text_column(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.strip()
    df[column_name] = df[column_name].str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    return df

def remove_duplicates(df, subset=None, keep='first'):
    return df.drop_duplicates(subset=subset, keep=keep)

def standardize_dates(df, column_name, date_format='%Y-%m-%d'):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    df[column_name] = df[column_name].dt.strftime(date_format)
    return df

def clean_dataset(df, text_columns=None, date_columns=None, deduplicate=True):
    if text_columns:
        for col in text_columns:
            df = clean_text_column(df, col)
    
    if date_columns:
        for col in date_columns:
            df = standardize_dates(df, col)
    
    if deduplicate:
        df = remove_duplicates(df)
    
    return df
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = df[column].fillna(df[column].median())
    
    # Remove outliers using z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    for column in numeric_cols:
        if df[column].std() != 0:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Clean CSV data by handling missing values and converting data types.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Save cleaned data
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"cleaned_{input_file.name}"
        
        df.to_csv(output_path, index=False)
        
        # Print summary
        print(f"Data cleaning completed:")
        print(f"  - Original file: {input_path}")
        print(f"  - Cleaned file: {output_path}")
        print(f"  - Rows processed: {initial_rows}")
        print(f"  - Duplicates removed: {duplicates_removed}")
        print(f"  - Missing values filled: {df.isnull().sum().sum()}")
        
        return df, output_path
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty at {input_path}")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None, None

def validate_dataframe(df):
    """
    Validate dataframe structure and content.
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return True, validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Alice'],
        'age': [25, 30, None, 35, 28, 25],
        'salary': [50000, 60000, 55000, None, 52000, 50000],
        'join_date': ['2020-01-15', '2019-03-20', None, '2021-07-10', '2020-11-05', '2020-01-15']
    }
    
    # Create sample CSV
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df, output_file = clean_csv_data('sample_data.csv')
    
    if cleaned_df is not None:
        # Validate cleaned data
        is_valid, validation_info = validate_dataframe(cleaned_df)
        print(f"\nData validation: {'Passed' if is_valid else 'Failed'}")
        if is_valid:
            for key, value in validation_info.items():
                print(f"  {key}: {value}")