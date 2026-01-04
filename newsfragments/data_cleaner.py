import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    for col in numeric_columns:
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in the dataset")
    
    return True
import pandas as pd

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicate rows and
    filling missing numeric values with column median.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values with column median
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(
        df_cleaned[numeric_cols].median()
    )
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validate DataFrame structure and content.
    Returns True if valid, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    return True

def process_csv_file(input_path, output_path):
    """
    Read CSV file, clean the data, and save to output path.
    """
    try:
        df = pd.read_csv(input_path)
        
        if not validate_dataframe(df):
            raise ValueError("Invalid DataFrame structure")
        
        df_cleaned = clean_dataframe(df)
        df_cleaned.to_csv(output_path, index=False)
        
        print(f"Data cleaned successfully. Saved to {output_path}")
        print(f"Original rows: {len(df)}, Cleaned rows: {len(df_cleaned)}")
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    process_csv_file(input_file, output_file)import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing specified string columns.
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # If no columns specified, clean all object (string) columns
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns
    
    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(_normalize_string)
    
    return df_clean

def _normalize_string(text):
    """
    Normalize a string: lowercase, strip whitespace, remove extra spaces.
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    text = text.lower()
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    Returns a Series with boolean values.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].apply(lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False)
import pandas as pd
import re

def clean_text_column(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    return df

def remove_duplicates(df, subset=None):
    initial_count = len(df)
    df_cleaned = df.drop_duplicates(subset=subset, keep='first')
    removed_count = initial_count - len(df_cleaned)
    return df_cleaned, removed_count

def standardize_dates(df, date_column, target_format='%Y-%m-%d'):
    try:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df[date_column] = df[date_column].dt.strftime(target_format)
    except Exception as e:
        print(f"Date standardization failed: {e}")
    return df

def validate_email_column(df, email_column):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].str.match(pattern, na=False)
    return df

def process_dataframe(df, text_columns=None, date_columns=None, email_columns=None):
    if text_columns:
        for col in text_columns:
            df = clean_text_column(df, col)
    
    df, duplicates_removed = remove_duplicates(df)
    
    if date_columns:
        for col in date_columns:
            df = standardize_dates(df, col)
    
    if email_columns:
        for col in email_columns:
            df = validate_email_column(df, col)
    
    return df, duplicates_removed