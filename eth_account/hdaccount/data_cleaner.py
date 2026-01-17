
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    """
    cleaned_df = df.copy()
    
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if columns_to_clean is None:
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns
    
    for col in columns_to_clean:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    return cleaned_df

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s-]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column and return a boolean Series.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].astype(str).str.match(email_pattern)import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")
        
        for col in df.select_dtypes(include=[np.number]).columns:
            df = remove_outliers_iqr(df, col)
        
        print(f"After outlier removal: {df.shape}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df = normalize_column(df, col)
        
        cleaned_file = file_path.replace('.csv', '_cleaned.csv')
        df.to_csv(cleaned_file, index=False)
        print(f"Cleaned data saved to: {cleaned_file}")
        return df
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200)
    })
    sample_data.to_csv('sample_data.csv', index=False)
    cleaned_df = clean_dataset('sample_data.csv')import numpy as np
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

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    if std_val == 0:
        return df[column]
    return (df[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='standard'):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
        if outlier_removal:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        if normalization == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalization == 'standard':
            cleaned_df[col] = standardize_zscore(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)