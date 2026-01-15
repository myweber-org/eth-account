
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_dataset(df, columns_to_clean=None):
    """
    Clean a dataset by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns_to_clean (list, optional): List of column names to clean.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_clean = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 200],
        'C': ['a', 'b', 'c', 'd', 'e', 'f']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, ['A', 'B'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_value: Value to use for filling missing data.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that the DataFrame contains all required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if all required columns are present, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers from a specific column using z-score method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to check for outliers.
    threshold (float): Z-score threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    from scipy import stats
    import numpy as np
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    mask = z_scores < threshold
    
    return df[mask].reset_index(drop=True)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
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
    
    return filtered_df

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If columns is None, clean all numeric columns.
    
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
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

def get_data_summary(df):
    """
    Generate summary statistics for a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    summary = {
        'original_rows': len(df),
        'cleaned_rows': None,
        'removed_rows': None,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 10 + 50,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[0, 'value'] = 200
    df.loc[1, 'value'] = -100
    
    print("Original data shape:", df.shape)
    print("Original data summary:")
    print(df['value'].describe())
    
    cleaned_df = clean_numeric_data(df, ['value'])
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Cleaned data summary:")
    print(cleaned_df['value'].describe())
    
    summary = get_data_summary(df)
    print(f"\nRemoved {len(df) - len(cleaned_df)} outliers")
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, columns=None):
    """
    Normalize data using min-max scaling.
    
    Args:
        data: pandas DataFrame
        columns: list of columns to normalize (default: all numeric columns)
    
    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_data = data.copy()
    
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            col_min = data[col].min()
            col_max = data[col].max()
            
            if col_max != col_min:
                normalized_data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                normalized_data[col] = 0
    
    return normalized_data

def standardize_zscore(data, columns=None):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        columns: list of columns to standardize (default: all numeric columns)
    
    Returns:
        Standardized DataFrame
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    standardized_data = data.copy()
    
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            col_mean = data[col].mean()
            col_std = data[col].std()
            
            if col_std > 0:
                standardized_data[col] = (data[col] - col_mean) / col_std
            else:
                standardized_data[col] = 0
    
    return standardized_data

def clean_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (default: all columns)
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = data.columns.tolist()
    
    cleaned_data = data.copy()
    
    for col in columns:
        if col not in cleaned_data.columns:
            continue
            
        if cleaned_data[col].isnull().any():
            if strategy == 'drop':
                cleaned_data = cleaned_data.dropna(subset=[col])
            elif strategy == 'mean' and np.issubdtype(cleaned_data[col].dtype, np.number):
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mean())
            elif strategy == 'median' and np.issubdtype(cleaned_data[col].dtype, np.number):
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
            elif strategy == 'mode':
                mode_value = cleaned_data[col].mode()
                if not mode_value.empty:
                    cleaned_data[col] = cleaned_data[col].fillna(mode_value.iloc[0])
                else:
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].iloc[0])
            else:
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].iloc[0])
    
    return cleaned_data

def validate_dataframe(data, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        data: pandas DataFrame to validate
        required_columns: list of required column names
        numeric_columns: list of columns that must be numeric
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if numeric_columns:
        non_numeric_cols = [col for col in numeric_columns 
                           if col in data.columns and not np.issubdtype(data[col].dtype, np.number)]
        if non_numeric_cols:
            return False, f"Non-numeric columns found: {non_numeric_cols}"
    
    return True, "DataFrame validation passed"