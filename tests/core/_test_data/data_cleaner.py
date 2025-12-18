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
    
    return filtered_df

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistical measures
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_df

def generate_cleaning_report(original_df, cleaned_df):
    """
    Generate a report comparing original and cleaned DataFrames.
    
    Args:
        original_df (pd.DataFrame): Original DataFrame
        cleaned_df (pd.DataFrame): Cleaned DataFrame
    
    Returns:
        pd.DataFrame: Report DataFrame
    """
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    
    report_data = []
    
    for col in numeric_cols:
        if col in cleaned_df.columns:
            original_stats = calculate_basic_stats(original_df, col)
            cleaned_stats = calculate_basic_stats(cleaned_df, col)
            
            report_data.append({
                'column': col,
                'original_rows': original_stats['count'],
                'cleaned_rows': cleaned_stats['count'],
                'rows_removed': original_stats['count'] - cleaned_stats['count'],
                'original_mean': original_stats['mean'],
                'cleaned_mean': cleaned_stats['mean'],
                'original_std': original_stats['std'],
                'cleaned_std': cleaned_stats['std']
            })
    
    report_df = pd.DataFrame(report_data)
    
    return report_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[np.random.choice(df.index, 50), 'A'] = np.random.uniform(500, 1000, 50)
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal statistics:")
    for col in df.columns:
        stats = calculate_basic_stats(df, col)
        print(f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    cleaned_df = clean_numeric_data(df)
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    for col in cleaned_df.columns:
        stats = calculate_basic_stats(cleaned_df, col)
        print(f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    report = generate_cleaning_report(df, cleaned_df)
    print("\nCleaning Report:")
    print(report)
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column values to range [0, 1]
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize column values using z-score
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
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
            elif strategy == 'drop':
                data_filled = data_filled.dropna(subset=[column])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_filled[column] = data_filled[column].fillna(fill_value)
    
    return data_filled

def clean_dataset(data, config):
    """
    Apply multiple cleaning operations based on configuration
    """
    cleaned_data = data.copy()
    
    for column, operations in config.items():
        if column not in cleaned_data.columns:
            continue
            
        for operation in operations:
            if operation['type'] == 'remove_outliers':
                cleaned_data = remove_outliers_iqr(
                    cleaned_data, 
                    column, 
                    multiplier=operation.get('multiplier', 1.5)
                )
            elif operation['type'] == 'normalize':
                cleaned_data[column] = normalize_minmax(cleaned_data, column)
            elif operation['type'] == 'standardize':
                cleaned_data[column] = standardize_zscore(cleaned_data, column)
            elif operation['type'] == 'fill_missing':
                cleaned_data = handle_missing_values(
                    cleaned_data,
                    strategy=operation.get('strategy', 'mean'),
                    columns=[column]
                )
    
    return cleaned_dataimport pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    If columns specified, only check those columns.
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def fill_missing_with_mean(df, columns):
    """
    Fill missing values in specified columns with column mean.
    """
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            df_filled[col] = df_filled[col].fillna(mean_val)
    return df_filled

def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method for a specific column.
    Returns boolean Series where True indicates an outlier.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def remove_outliers(df, column):
    """
    Remove outliers from DataFrame for a specific column using IQR method.
    """
    outliers_mask = detect_outliers_iqr(df, column)
    return df[~outliers_mask]

def standardize_column(df, column):
    """
    Standardize a column to have mean=0 and std=1.
    """
    df_standardized = df.copy()
    if column in df.columns:
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df_standardized[column] = (df[column] - mean_val) / std_val
    return df_standardized

def clean_dataset(df, missing_strategy='remove', outlier_columns=None):
    """
    Comprehensive data cleaning function.
    missing_strategy: 'remove' or 'mean'
    outlier_columns: list of columns to remove outliers from
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df)
    elif missing_strategy == 'mean':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df = fill_missing_with_mean(cleaned_df, numeric_cols)
    
    # Remove outliers
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers(cleaned_df, col)
    
    return cleaned_df
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_df)
        print(f"Removed {removed_duplicates} duplicate rows.")
    
    if fill_missing:
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(cleaned_df[numeric_columns].mean())
        
        categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
        cleaned_df[categorical_columns] = cleaned_df[categorical_columns].fillna('Unknown')
        
        print("Missing values have been filled.")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate the DataFrame structure and content.
    """
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty.")
    
    print("DataFrame validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 90.0, 90.0, 78.5, None, 95.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    except ValueError as e:
        print(f"Validation error: {e}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
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
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return Trueimport pandas as pd

def clean_dataset(df):
    """
    Remove null values and duplicate rows from a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with no nulls or duplicates.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = df.copy()
    
    cleaned_df = cleaned_df.dropna()
    
    cleaned_df = cleaned_df.drop_duplicates()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_clean_dataset(df):
    """
    Validate that a DataFrame has no null values or duplicate rows.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        bool: True if DataFrame is clean, False otherwise.
    """
    if df.isnull().sum().sum() > 0:
        return False
    
    if df.duplicated().sum() > 0:
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, None, 4, 1],
        'B': [5, 6, 7, None, 5],
        'C': [8, 9, 10, 11, 8]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nNull values:", df.isnull().sum().sum())
    print("Duplicates:", df.duplicated().sum())
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nValidation:", validate_clean_dataset(cleaned))