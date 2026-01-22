
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_column(df: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize specified column using different methods.
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax', 'zscore', 'log')
    
    Returns:
        DataFrame with normalized column
    """
    df = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df[column] = (df[column] - mean_val) / std_val
    
    elif method == 'log':
        if df[column].min() > 0:
            df[column] = np.log(df[column])
    
    return df

def clean_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    
    elif strategy == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    
    elif strategy == 'mode':
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
    
    elif strategy == 'drop':
        df = df.dropna()
    
    return df

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if df.empty:
        return False
    
    if df.isnull().all().any():
        return False
    
    return True

def process_data_pipeline(df: pd.DataFrame, 
                         remove_dups: bool = True,
                         clean_missing: bool = True,
                         normalize_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Complete data processing pipeline.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicates
        clean_missing: Whether to clean missing values
        normalize_cols: List of columns to normalize
    
    Returns:
        Processed DataFrame
    """
    processed_df = df.copy()
    
    if not validate_dataframe(processed_df):
        raise ValueError("Invalid input DataFrame")
    
    if remove_dups:
        processed_df = remove_duplicates(processed_df)
    
    if clean_missing:
        processed_df = clean_missing_values(processed_df, strategy='mean')
    
    if normalize_cols:
        for col in normalize_cols:
            if col in processed_df.columns:
                processed_df = normalize_column(processed_df, col, method='minmax')
    
    return processed_df
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: Input pandas DataFrame
        column_mapping: Dictionary mapping old column names to new ones
        drop_duplicates: Whether to remove duplicate rows
        normalize_text: Whether to normalize text columns (strip, lower case)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df: Input pandas DataFrame
        email_column: Name of the column containing email addresses
    
    Returns:
        DataFrame with validation results
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validation_results = df.copy()
    validation_results['email_valid'] = validation_results[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    valid_count = validation_results['email_valid'].sum()
    total_count = len(validation_results)
    
    print(f"Email validation results: {valid_count}/{total_count} valid emails")
    
    return validation_results

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
        format: File format ('csv', 'excel', 'json')
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data saved to {output_path} in {format} format")import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, overwrites input file
        subset (list, optional): Columns to consider for identifying duplicates
        keep (str): Which duplicate to keep - 'first', 'last', or False to drop all
        
    Returns:
        int: Number of duplicates removed
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        final_rows = len(df_clean)
        
        duplicates_removed = initial_rows - final_rows
        
        if output_file is None:
            output_file = input_file
            
        df_clean.to_csv(output_file, index=False)
        
        print(f"Removed {duplicates_removed} duplicate rows")
        print(f"Original rows: {initial_rows}, Cleaned rows: {final_rows}")
        print(f"Saved to: {output_file}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)