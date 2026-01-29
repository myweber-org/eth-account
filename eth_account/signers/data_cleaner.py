
import pandas as pd
import re

def clean_dataframe(df, text_columns=None):
    """
    Clean a DataFrame by removing duplicate rows and standardizing text in specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        text_columns (list, optional): List of column names containing text to standardize.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = cleaned_df.shape[0]
    cleaned_df = cleaned_df.drop_duplicates()
    removed_duplicates = initial_rows - cleaned_df.shape[0]
    
    # Standardize text in specified columns
    if text_columns:
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].apply(_standardize_text)
    
    return cleaned_df

def _standardize_text(text):
    """
    Standardize text by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        text (str): Input text to standardize.
    
    Returns:
        str: Standardized text.
    """
    if not isinstance(text, str):
        return text
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters except alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with additional 'email_valid' column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    validated_df = df.copy()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validated_df['email_valid'] = validated_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    return validated_dfimport pandas as pd

def clean_dataset(df):
    """
    Remove null values and duplicate rows from a pandas DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame with nulls removed and duplicates dropped.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validate that the input is a pandas DataFrame and not empty.
    
    Parameters:
    df: Input to validate
    
    Returns:
    bool: True if valid, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input must be a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = pd.DataFrame({
#         'A': [1, 2, None, 4, 2],
#         'B': [5, 6, 7, None, 6],
#         'C': [8, 9, 10, 11, 9]
#     })
#     
#     if validate_dataframe(sample_data):
#         cleaned_data = clean_dataset(sample_data)
#         print("Original shape:", sample_data.shape)
#         print("Cleaned shape:", cleaned_data.shape)
#         print("\nCleaned data:")
#         print(cleaned_data)
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean dataset by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for missing values - 'mean', 'median', 'mode', or 'drop'
    outlier_method (str): Method for outlier detection - 'iqr' or 'zscore'
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    else:
        numeric_cols = [col for col in columns if col in df_clean.columns]
    
    for col in numeric_cols:
        if missing_strategy != 'drop':
            if missing_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif missing_strategy == 'median':
                fill_value = df_clean[col].median()
            elif missing_strategy == 'mode':
                fill_value = df_clean[col].mode()[0]
            else:
                fill_value = 0
            
            df_clean[col].fillna(fill_value, inplace=True)
        else:
            df_clean = df_clean.dropna(subset=[col])
    
    if outlier_method == 'iqr':
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean[col] = np.where(
                (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound),
                df_clean[col].median(),
                df_clean[col]
            )
    
    elif outlier_method == 'zscore':
        for col in numeric_cols:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean[col] = np.where(
                z_scores > 3,
                df_clean[col].median(),
                df_clean[col]
            )
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "Dataframe is empty"
    
    if len(df) < min_rows:
        return False, f"Dataframe has less than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def get_data_summary(df):
    """
    Generate summary statistics for dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data = clean_dataset(sample_data, missing_strategy='median', outlier_method='iqr')
    print("\nCleaned data:")
    print(cleaned_data)
    
    is_valid, message = validate_data(cleaned_data, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")
    
    summary = get_data_summary(cleaned_data)
    print(f"\nData shape: {summary['shape']}")
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            mode_value = cleaned_df[col].mode()
            if not mode_value.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_value.iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()

    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate row(s).")

    if cleaned_df.isnull().sum().any():
        print("Handling missing values...")
        if fill_missing == 'mean':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        elif fill_missing == 'median':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        elif fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
        else:
            raise ValueError("fill_missing must be 'mean', 'median', or 'drop'")
        print("Missing values handled.")

    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, np.nan],
        'B': [5, np.nan, 5, 8, 9],
        'C': ['x', 'y', 'x', 'z', 'z']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (drop duplicates, fill with mean):")
    cleaned = clean_dataframe(df, drop_duplicates=True, fill_missing='mean')
    print(cleaned)