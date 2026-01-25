import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].copy()
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].copy()
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_filled = data.copy()
    
    for column in columns:
        if column not in data.columns:
            continue
            
        if data[column].isnull().any():
            if strategy == 'mean':
                fill_value = data[column].mean()
            elif strategy == 'median':
                fill_value = data[column].median()
            elif strategy == 'mode':
                fill_value = data[column].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_filled[column] = data[column].fillna(fill_value)
    
    return data_filled

def create_data_summary(data):
    """
    Create a summary statistics DataFrame
    """
    summary = pd.DataFrame({
        'count': data.count(),
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        '25%': data.quantile(0.25),
        '50%': data.quantile(0.50),
        '75%': data.quantile(0.75),
        'max': data.max(),
        'missing': data.isnull().sum(),
        'missing_pct': (data.isnull().sum() / len(data)) * 100
    })
    
    return summary.T
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_mapping (dict): Optional dictionary to rename columns
    drop_duplicates (bool): Whether to remove duplicate rows
    normalize_text (bool): Whether to normalize text columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].apply(lambda x: normalize_string(x) if pd.notnull(x) else x)
    
    return cleaned_df

def normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Parameters:
    text (str): Input string
    
    Returns:
    str: Normalized string
    """
    if not isinstance(text, str):
        return text
    
    normalized = text.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    
    return normalized

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Parameters:
    email (str): Email address to validate
    
    Returns:
    bool: True if email is valid, False otherwise
    """
    if not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

if __name__ == "__main__":
    sample_data = {
        'Name': ['John Doe', 'Jane Smith', 'JOHN DOE', 'Bob Johnson  ', 'Alice Brown'],
        'Email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.com', 'alice@domain.net'],
        'Age': [25, 30, 25, 150, 28],
        'Score': [85.5, 92.0, 85.5, 45.2, 78.9]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataframe(df, drop_duplicates=True, normalize_text=True)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    valid_emails = cleaned['Email'].apply(validate_email)
    print("Valid emails:")
    print(valid_emails)
    print("\n" + "="*50 + "\n")
    
    filtered = remove_outliers_iqr(cleaned, 'Age')
    print("DataFrame after removing age outliers:")
    print(filtered)import pandas as pd
import numpy as np

def remove_duplicates(df):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    """
    Fill missing values using specified strategy.
    """
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.fillna(0)

def normalize_column(df, column_name):
    """
    Normalize specified column to range [0,1].
    """
    if column_name in df.columns:
        col = df[column_name]
        df[column_name] = (col - col.min()) / (col.max() - col.min())
    return df

def filter_outliers(df, column_name, threshold=3):
    """
    Remove outliers based on z-score threshold.
    """
    if column_name in df.columns:
        z_scores = np.abs((df[column_name] - df[column_name].mean()) / df[column_name].std())
        return df[z_scores < threshold]
    return df

def clean_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations sequentially.
    """
    if operations is None:
        operations = ['remove_duplicates', 'fill_missing']
    
    if 'remove_duplicates' in operations:
        df = remove_duplicates(df)
    
    if 'fill_missing' in operations:
        df = fill_missing_values(df)
    
    return df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None, 7],
        'B': [10, 20, 20, 40, 50, 60, 1000]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)