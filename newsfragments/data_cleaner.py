
import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean a CSV file by handling missing values.
    
    Parameters:
    filepath (str): Path to the CSV file.
    fill_strategy (str): Method for filling missing values ('mean', 'median', 'mode', or 'zero').
    drop_threshold (float): Drop columns with missing ratio above this threshold (0.0 to 1.0).
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    # Calculate missing ratio per column
    missing_ratio = df.isnull().sum() / len(df)
    
    # Drop columns with high missing ratio
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    if len(columns_to_drop) > 0:
        print(f"Dropped columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
    
    # Fill remaining missing values based on strategy
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'mean':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
    elif fill_strategy == 'median':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    elif fill_strategy == 'mode':
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    elif fill_strategy == 'zero':
        df = df.fillna(0)
    else:
        raise ValueError("fill_strategy must be 'mean', 'median', 'mode', or 'zero'")
    
    # For categorical columns, fill with most frequent value
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    final_shape = df.shape
    print(f"Cleaned data shape: {final_shape}")
    print(f"Removed {original_shape[1] - final_shape[1]} columns, {original_shape[0] - final_shape[0]} rows")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Parameters:
    df (pandas.DataFrame): Cleaned DataFrame.
    output_path (str): Path for the output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, fill_strategy='median', drop_threshold=0.3)
        save_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")import pandas as pd

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
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def validate_dataframe(df):
    """
    Basic validation of DataFrame structure.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    if df.empty:
        return True
    return True

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    return cleaned_dfimport numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or array-like): The dataset.
    column (int or str): The column index or name to process.
    
    Returns:
    tuple: (cleaned_data, outliers_removed)
    """
    if isinstance(column, str):
        raise ValueError("Column name handling requires pandas DataFrame. Use integer index for lists/arrays.")
    
    data_array = np.array(data)
    column_data = data_array[:, column].astype(float)
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    cleaned_data = data_array[mask]
    outliers = data_array[~mask]
    
    return cleaned_data, outliers

def example_usage():
    sample_data = [
        [1, 150.5],
        [2, 200.2],
        [3, 50.1],
        [4, 300.9],
        [5, 180.3],
        [6, 1000.0],
        [7, 190.7]
    ]
    
    cleaned, removed = remove_outliers_iqr(sample_data, column=1)
    print(f"Original data points: {len(sample_data)}")
    print(f"Cleaned data points: {len(cleaned)}")
    print(f"Outliers removed: {len(removed)}")
    print(f"Outliers: {removed}")

if __name__ == "__main__":
    example_usage()
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

if __name__ == "__main__":
    sample_data = {'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 12, 10, 9, 8, 15, 200]}
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\nOriginal Statistics:")
    print(calculate_summary_statistics(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Statistics:")
    print(calculate_summary_statistics(cleaned_df, 'values'))