import pandas as pd
import numpy as np

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

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
        df = normalize_minmax(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('sample_data.csv')
    cleaned_df.to_csv('cleaned_data.csv', index=False)
    print("Data cleaning completed. Saved to cleaned_data.csv")
import pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.5):
    """
    Remove rows with missing values exceeding threshold percentage.
    
    Args:
        df (pd.DataFrame): Input dataframe
        threshold (float): Maximum allowed missing percentage per row (0-1)
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    missing_percentage = df.isnull().mean(axis=1)
    return df[missing_percentage <= threshold].copy()

def fill_missing_with_median(df, columns=None):
    """
    Fill missing values with column median.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): Specific columns to fill, None for all numeric columns
    
    Returns:
        pd.DataFrame: Dataframe with filled values
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            median_val = df[col].median()
            df_filled[col] = df[col].fillna(median_val)
    
    return df_filled

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def cap_outliers(df, column, method='iqr', percentile_low=1, percentile_high=99):
    """
    Cap outliers to specified percentiles or IQR bounds.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to process
        method (str): 'iqr' or 'percentile' method
        percentile_low (float): Lower percentile for capping
        percentile_high (float): Upper percentile for capping
    
    Returns:
        pd.DataFrame: Dataframe with capped values
    """
    df_capped = df.copy()
    
    if method == 'iqr':
        outliers = detect_outliers_iqr(df, column)
        if outliers.any():
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_capped.loc[df[column] < lower_bound, column] = lower_bound
            df_capped.loc[df[column] > upper_bound, column] = upper_bound
    
    elif method == 'percentile':
        lower_val = df[column].quantile(percentile_low / 100)
        upper_val = df[column].quantile(percentile_high / 100)
        
        df_capped.loc[df[column] < lower_val, column] = lower_val
        df_capped.loc[df[column] > upper_val, column] = upper_val
    
    return df_capped

def normalize_column(df, column, method='minmax'):
    """
    Normalize column values.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to normalize
        method (str): 'minmax' or 'zscore' normalization
    
    Returns:
        pd.DataFrame: Dataframe with normalized column
    """
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    
    return df_normalized

def clean_dataframe(df, missing_threshold=0.3, outlier_columns=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input dataframe
        missing_threshold (float): Threshold for removing rows with missing values
        outlier_columns (list): Columns to check for outliers
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    # Remove rows with excessive missing values
    cleaned_df = remove_missing_rows(cleaned_df, threshold=missing_threshold)
    
    # Fill remaining missing values
    cleaned_df = fill_missing_with_median(cleaned_df)
    
    # Cap outliers if specified
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = cap_outliers(cleaned_df, col, method='iqr')
    
    return cleaned_df