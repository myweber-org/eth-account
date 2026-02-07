
import pandas as pd

def clean_dataset(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_mapping (dict, optional): Dictionary mapping old column names to new ones.
        drop_duplicates (bool): Whether to remove duplicate rows.
        normalize_text (bool): Whether to normalize text columns (strip, lower case).
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Rename columns if mapping provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Remove duplicate rows
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    # Normalize text columns
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    # Convert date columns if they exist
    date_columns = cleaned_df.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['John Doe', 'Jane Smith', 'John Doe', ' Bob Johnson '],
        'Email': ['john@example.com', 'jane@example.com', 'john@example.com', 'bob@example.com'],
        'Age': [25, 30, 25, 35],
        'Join_Date': ['2023-01-01', '2023-02-01', '2023-01-01', '2023-03-01']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, column_mapping={'Join_Date': 'join_date'})
    print("Cleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, required_columns=['name', 'email', 'age'])
    print(f"\nValidation: {is_valid} - {message}")
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: Input pandas DataFrame
        column_mapping: Dictionary to rename columns (old_name: new_name)
        drop_duplicates: Boolean to remove duplicate rows
        normalize_text: Boolean to normalize text columns (strip, lower, remove extra spaces)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for column in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[column] = cleaned_df[column].astype(str).apply(
                lambda x: re.sub(r'\s+', ' ', x.strip().lower())
            )
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df: Input pandas DataFrame
        email_column: Name of the column containing email addresses
    
    Returns:
        DataFrame with additional 'email_valid' boolean column
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validated_df = df.copy()
    validated_df['email_valid'] = validated_df[email_column].str.match(email_pattern)
    
    return validated_df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a numeric column using the Interquartile Range method.
    
    Args:
        df: Input pandas DataFrame
        column: Name of the numeric column
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Clean CSV data by handling missing values and standardizing formats.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Remove duplicate rows
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Remove leading/trailing whitespace from string columns
        for col in categorical_cols:
            df[col] = df[col].astype(str).str.strip()
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"cleaned_{input_file.name}"
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        
        # Generate summary statistics
        summary = {
            'original_rows': initial_count,
            'cleaned_rows': len(df),
            'duplicates_removed': duplicates_removed,
            'numeric_cols_filled': len([col for col in numeric_cols if df[col].isnull().any()]),
            'categorical_cols_filled': len([col for col in categorical_cols if df[col].isnull().any()]),
            'output_file': str(output_path)
        }
        
        return df, summary
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None, None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None, None

def validate_dataframe(df):
    """
    Validate dataframe for common data quality issues.
    """
    if df is None or df.empty:
        return {'valid': False, 'message': 'DataFrame is empty or None'}
    
    validation_results = {
        'valid': True,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict()
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
        'Age': [25, None, 30, 35, 40],
        'City': ['New York', 'London', None, 'Paris', 'Tokyo'],
        'Score': [85.5, 92.0, 78.5, None, 88.0]
    })
    
    # Save sample data
    sample_data.to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df, summary = clean_csv_data('sample_data.csv')
    
    if cleaned_df is not None:
        print("Data cleaning completed successfully")
        print(f"Summary: {summary}")
        
        # Validate cleaned data
        validation = validate_dataframe(cleaned_df)
        print(f"Validation results: {validation}")
        
        # Display cleaned data
        print("\nCleaned Data:")
        print(cleaned_df.head())
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'drop',
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and standardizing column names.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        missing_strategy: Strategy for handling missing values ('drop', 'fill', 'interpolate')
        fill_value: Value to fill missing data with when using 'fill' strategy
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(input_path)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Handle missing values
        if missing_strategy == 'drop':
            df = df.dropna()
        elif missing_strategy == 'fill':
            if fill_value is not None:
                df = df.fillna(fill_value)
            else:
                df = df.fillna(df.mean(numeric_only=True))
        elif missing_strategy == 'interpolate':
            df = df.interpolate(method='linear', limit_direction='forward')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        
        print(f"Data cleaning completed. Cleaned data saved to: {output_path}")
        print(f"Original shape: {df.shape}, Cleaned shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        raise
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        raise

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame and return summary statistics.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'date_columns': list(df.select_dtypes(include=['datetime64']).columns)
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_df = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Value': [10.5, np.nan, 15.2, 15.2, 20.1],
        'Category': ['A', 'B', 'A', 'B', 'C']
    })
    
    # Save sample data
    sample_df.to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data(
        input_path='sample_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='fill',
        fill_value=0.0
    )
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned_df)
    print("Validation results:", validation)