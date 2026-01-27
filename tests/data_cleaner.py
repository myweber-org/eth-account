import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', remove_duplicates=True):
    """
    Load and clean CSV data by handling missing values and duplicates.
    
    Args:
        filepath (str): Path to the CSV file.
        fill_method (str): Method for filling missing values ('mean', 'median', 'mode').
        remove_duplicates (bool): Whether to remove duplicate rows.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    original_shape = df.shape
    
    if remove_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")
    
    if df.isnull().sum().any():
        print("Handling missing values...")
        for column in df.select_dtypes(include=[np.number]).columns:
            if df[column].isnull().any():
                if fill_method == 'mean':
                    fill_value = df[column].mean()
                elif fill_method == 'median':
                    fill_value = df[column].median()
                elif fill_method == 'mode':
                    fill_value = df[column].mode()[0]
                else:
                    fill_value = 0
                
                df[column].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{column}' with {fill_method}: {fill_value}")
    
    print(f"Data cleaning complete. Original shape: {original_shape}, Cleaned shape: {df.shape}")
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame.
        output_path (str): Path for the output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, fill_method='median')
        save_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in specified columns using a given strategy.
    """
    df_filled = df.copy()
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            df_filled[col].fillna(fill_value, inplace=True)
        else:
            df_filled[col].fillna('Unknown', inplace=True)
    return df_filled

def normalize_column(df, column):
    """
    Normalize a numeric column to range [0, 1].
    """
    if df[column].dtype in [np.float64, np.int64]:
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a column using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataframe(df, operations):
    """
    Apply a sequence of cleaning operations to a DataFrame.
    operations: list of tuples (function_name, kwargs)
    """
    cleaned_df = df.copy()
    for func_name, kwargs in operations:
        if func_name == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, **kwargs)
        elif func_name == 'fill_missing':
            cleaned_df = fill_missing_values(cleaned_df, **kwargs)
        elif func_name == 'normalize':
            cleaned_df = normalize_column(cleaned_df, **kwargs)
        elif func_name == 'remove_outliers':
            cleaned_df = remove_outliers_iqr(cleaned_df, **kwargs)
    return cleaned_df