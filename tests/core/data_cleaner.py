
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 50, 10)
        ]),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original dataset shape: {df.shape}")
    
    cleaned_df = clean_dataset(df, ['value'])
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Outliers removed: {len(df) - len(cleaned_df)}")
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_clean (list, optional): List of column names to apply string normalization.
                                       If None, applies to all object dtype columns.
    remove_duplicates (bool): If True, remove duplicate rows.
    case_normalization (str): One of 'lower', 'upper', or None for case normalization.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows.")
    
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str)
            
            if case_normalization == 'lower':
                df_clean[col] = df_clean[col].str.lower()
            elif case_normalization == 'upper':
                df_clean[col] = df_clean[col].str.upper()
            
            df_clean[col] = df_clean[col].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    
    return df_clean

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    email_column (str): Name of the column containing email addresses.
    
    Returns:
    pd.DataFrame: DataFrame with additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].str.match(email_pattern, na=False)
    
    valid_count = df['email_valid'].sum()
    total_count = len(df)
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'ALICE BROWN', '  Bob   White  '],
        'email': ['john@example.com', 'jane@test.org', 'john@example.com', 'invalid-email', 'bob@company.co.uk'],
        'age': [25, 30, 25, 35, 40]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataframe(df, case_normalization='lower')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    validated_df = validate_email_column(cleaned_df, 'email')
    print("DataFrame with email validation:")
    print(validated_df)