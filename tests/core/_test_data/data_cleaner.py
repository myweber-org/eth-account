
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('input_data.csv', ['age', 'salary', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Original shape: {pd.read_csv('input_data.csv').shape}, Cleaned shape: {cleaned_data.shape}")
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_mapping (dict, optional): Dictionary mapping original column names to new names.
        drop_duplicates (bool): Whether to remove duplicate rows.
        normalize_text (bool): Whether to normalize text columns (strip, lower case, remove extra spaces).
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x.strip().lower()))
    
    return cleaned_df

def validate_email(email_string):
    """
    Validate email format using regex.
    
    Args:
        email_string (str): Email address to validate.
    
    Returns:
        bool: True if email format is valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email_string)))

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): Path to save the file.
        format (str): File format ('csv', 'excel', 'json').
    
    Raises:
        ValueError: If format is not supported.
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'excel', or 'json'.")

if __name__ == "__main__":
    sample_data = {
        'Name': [' John Doe ', 'Jane Smith', 'John Doe', 'ALICE BROWN  '],
        'Email': ['john@example.com', 'invalid-email', 'JOHN@example.com', 'alice@test.org'],
        'Age': [25, 30, 25, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df, column_mapping={'Name': 'full_name'})
    cleaned['email_valid'] = cleaned['Email'].apply(validate_email)
    
    print("\nCleaned DataFrame:")
    print(cleaned)