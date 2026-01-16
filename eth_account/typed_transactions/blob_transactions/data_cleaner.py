import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
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
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fillna_strategy='mean', columns_to_clean=None):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fillna_strategy (str): Strategy for filling NaN values ('mean', 'median', 'mode', or 'drop').
    columns_to_clean (list): Specific columns to apply cleaning. If None, all columns are cleaned.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if columns_to_clean is None:
        columns_to_clean = df_clean.columns.tolist()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    for col in columns_to_clean:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].dtype in [np.float64, np.int64]:
            if fillna_strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif fillna_strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif fillna_strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif fillna_strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
        else:
            if fillna_strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif fillna_strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
            else:
                df_clean[col].fillna('Unknown', inplace=True)
    
    return df_clean

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a specific column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): Columns to standardize. If None, all numeric columns are used.
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns.
    """
    df_std = df.copy()
    
    if columns is None:
        numeric_cols = df_std.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col in df_std.columns and df_std[col].dtype in [np.float64, np.int64]:
            mean = df_std[col].mean()
            std = df_std[col].std()
            if std > 0:
                df_std[col] = (df_std[col] - mean) / std
    
    return df_std

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, np.nan, 7],
        'B': [10, 20, 20, 40, 50, 60, 70],
        'C': ['x', 'y', 'y', 'z', np.nan, 'x', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, fillna_strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    standardized_df = standardize_columns(cleaned_df, columns=['A', 'B'])
    print("\nStandardized DataFrame:")
    print(standardized_df)import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    return df.drop_duplicates(subset=subset, keep=keep)

def clean_numeric_column(series, fill_method='mean'):
    """
    Clean a numeric series by filling missing values.

    Args:
        series (pd.Series): Input series.
        fill_method (str): Method to fill missing values.

    Returns:
        pd.Series: Cleaned series.
    """
    if series.isnull().all():
        return series
    if fill_method == 'mean':
        fill_value = series.mean()
    elif fill_method == 'median':
        fill_value = series.median()
    else:
        fill_value = 0
    return series.fillna(fill_value)