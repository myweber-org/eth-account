
import pandas as pd
import numpy as np

def remove_outliers(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return df[z_scores < threshold]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, output_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)
import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase and removing extra whitespace.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_email_column(df, column_name):
    """
    Validate email format in a column and return a boolean mask.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[column_name].astype(str).str.match(email_pattern)

def clean_dataset(df, text_columns=None, deduplicate=True, email_columns=None):
    """
    Perform comprehensive cleaning on a dataset.
    """
    cleaned_df = df.copy()
    
    if text_columns:
        for col in text_columns:
            cleaned_df = clean_text_column(cleaned_df, col)
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if email_columns:
        for col in email_columns:
            mask = validate_email_column(cleaned_df, col)
            cleaned_df = cleaned_df[mask].reset_index(drop=True)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'john doe', 'Bob Johnson  '],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.net'],
        'age': [25, 30, 25, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    
    cleaned = clean_dataset(
        df,
        text_columns=['name'],
        deduplicate=True,
        email_columns=['email']
    )
    print(cleaned)