
import pandas as pd
import numpy as np

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by standardizing columns, removing duplicates,
    and handling missing values.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in ['int64', 'float64']:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            elif cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_data(df, required_columns=None, unique_constraints=None):
    """
    Validate DataFrame structure and constraints.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if unique_constraints:
        for constraint in unique_constraints:
            if constraint in df.columns:
                if df[constraint].duplicated().any():
                    raise ValueError(f"Duplicate values found in unique constraint column: {constraint}")
    
    return True

def process_data_file(file_path, output_path=None, **clean_kwargs):
    """
    Load, clean, and optionally save a data file.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")
    
    cleaned_df = clean_dataframe(df, **clean_kwargs)
    
    if output_path:
        if output_path.endswith('.csv'):
            cleaned_df.to_csv(output_path, index=False)
        elif output_path.endswith('.xlsx'):
            cleaned_df.to_excel(output_path, index=False)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Alice', None, 'Charlie'],
        'Age': [25, 30, 25, 40, None],
        'City': ['NYC', 'LA', 'NYC', 'Chicago', 'Boston']
    })
    
    print("Original Data:")
    print(sample_data)
    print("\nCleaned Data:")
    cleaned = clean_dataframe(sample_data, drop_duplicates=True, fill_missing=True)
    print(cleaned)