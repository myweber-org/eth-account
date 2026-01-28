import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, processes all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10],
        'humidity': [45, 46, 47, 48, 49, 200, 50, 51, 52, -5],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning numeric columns...")
    
    cleaned_df = clean_numeric_data(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)import pandas as pd

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by standardizing columns, removing duplicates,
    and handling missing values.
    """
    cleaned_df = df.copy()
    
    # Rename columns if mapping is provided
    if column_mapping:
        cleaned_df.rename(columns=column_mapping, inplace=True)
    
    # Remove duplicate rows
    if drop_duplicates:
        cleaned_df.drop_duplicates(inplace=True)
    
    # Fill missing values
    if fill_missing:
        for column, value in fill_missing.items():
            if column in cleaned_df.columns:
                cleaned_df[column].fillna(value, inplace=True)
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    Returns a Series with boolean values indicating valid emails.
    """
    import re
    
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )

def standardize_dates(df, date_columns, date_format='%Y-%m-%d'):
    """
    Standardize date columns to a consistent format.
    """
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].dt.strftime(date_format)
    
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['John', 'Jane', 'John', 'Bob'],
        'Email': ['john@example.com', 'jane@example.com', 'john@example.com', 'invalid-email'],
        'Join_Date': ['2023-01-15', '2023-02-20', '2023-01-15', '2023-03-10']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    column_map = {'Join_Date': 'join_date'}
    cleaned = clean_dataset(
        df,
        column_mapping=column_map,
        drop_duplicates=True,
        fill_missing={'Email': 'unknown@example.com'}
    )
    
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    # Validate emails
    email_valid = validate_email_column(cleaned, 'Email')
    print("Valid emails:")
    print(email_valid)import numpy as np
import pandas as pd

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
    if max_val - min_val == 0:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.dropna()

def calculate_statistics(df):
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'median': df[col].median(),
            'skewness': df[col].skew()
        }
    return pd.DataFrame(stats).T

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(50, 200),
        'C': np.random.uniform(0, 1, 200)
    })
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'])
    print("Original shape:", sample_data.shape)
    print("Cleaned shape:", cleaned.shape)
    print("\nStatistics:")
    print(calculate_statistics(cleaned))