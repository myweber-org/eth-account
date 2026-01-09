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
    
    return True, "DataFrame is valid"
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def process_numerical_data(df, columns):
    """
    Process multiple numerical columns by removing outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to process
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    processed_df = df.copy()
    
    for col in columns:
        if col in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[col]):
            processed_df = remove_outliers_iqr(processed_df, col)
    
    return processed_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 200],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = process_numerical_data(df, ['temperature', 'humidity', 'pressure'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print()
    
    for col in ['temperature', 'humidity', 'pressure']:
        stats = calculate_summary_statistics(cleaned_df, col)
        print(f"Statistics for {col}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
        print()
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, columns=None, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    dataframe: pandas DataFrame
    columns: list of column names to process (default: all numeric columns)
    threshold: IQR multiplier for outlier detection
    
    Returns:
    Cleaned DataFrame with outliers removed
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    df_clean = dataframe.copy()
    
    for column in columns:
        if column in df_clean.columns:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = (df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(dataframe, columns=None, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    dataframe: pandas DataFrame
    columns: list of column names to normalize (default: all numeric columns)
    feature_range: tuple of (min, max) for scaled range
    
    Returns:
    DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    df_normalized = dataframe.copy()
    min_val, max_val = feature_range
    
    for column in columns:
        if column in df_normalized.columns:
            col_min = df_normalized[column].min()
            col_max = df_normalized[column].max()
            
            if col_max != col_min:
                df_normalized[column] = (df_normalized[column] - col_min) / (col_max - col_min)
                df_normalized[column] = df_normalized[column] * (max_val - min_val) + min_val
    
    return df_normalized

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect columns with skewed distributions.
    
    Parameters:
    dataframe: pandas DataFrame
    threshold: absolute skewness threshold for detection
    
    Returns:
    Dictionary with column names and their skewness values
    """
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    skewed_columns = {}
    
    for column in numeric_cols:
        skewness = stats.skew(dataframe[column].dropna())
        if abs(skewness) > threshold:
            skewed_columns[column] = skewness
    
    return skewed_columns

def log_transform_skewed(dataframe, skewed_columns):
    """
    Apply log transformation to reduce skewness.
    
    Parameters:
    dataframe: pandas DataFrame
    skewed_columns: dictionary from detect_skewed_columns function
    
    Returns:
    DataFrame with log-transformed columns
    """
    df_transformed = dataframe.copy()
    
    for column in skewed_columns.keys():
        if column in df_transformed.columns:
            min_val = df_transformed[column].min()
            if min_val <= 0:
                df_transformed[column] = np.log1p(df_transformed[column] - min_val + 1)
            else:
                df_transformed[column] = np.log(df_transformed[column])
    
    return df_transformed

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    
    Parameters:
    dataframe: pandas DataFrame
    strategy: 'mean', 'median', 'mode', or 'drop'
    columns: list of columns to process (default: all columns)
    
    Returns:
    DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns
    
    df_handled = dataframe.copy()
    
    for column in columns:
        if column in df_handled.columns:
            if strategy == 'drop':
                df_handled = df_handled.dropna(subset=[column])
            elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df_handled[column]):
                df_handled[column] = df_handled[column].fillna(df_handled[column].mean())
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_handled[column]):
                df_handled[column] = df_handled[column].fillna(df_handled[column].median())
            elif strategy == 'mode':
                df_handled[column] = df_handled[column].fillna(df_handled[column].mode()[0])
    
    return df_handled.reset_index(drop=True)