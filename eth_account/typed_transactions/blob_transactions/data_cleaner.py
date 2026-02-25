
import pandas as pd

def clean_dataframe(df, column, threshold, keep_above=True):
    """
    Filters a DataFrame based on a numeric threshold in a specified column.
    Returns a new DataFrame.
    """
    if keep_above:
        filtered_df = df[df[column] > threshold].copy()
    else:
        filtered_df = df[df[column] <= threshold].copy()

    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df

def remove_duplicates(df, subset=None):
    """
    Removes duplicate rows from the DataFrame.
    If subset is provided, only considers those columns for identifying duplicates.
    """
    cleaned_df = df.drop_duplicates(subset=subset, keep='first').copy()
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df

def main():
    # Example usage
    data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Alice'],
            'Score': [85, 92, 78, 45, 85],
            'Age': [25, 30, 35, 40, 25]}
    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)

    # Clean based on score threshold
    high_scorers = clean_dataframe(df, 'Score', 80, keep_above=True)
    print("\nDataFrame with scores > 80:")
    print(high_scorers)

    # Remove duplicates
    unique_df = remove_duplicates(df, subset=['Name', 'Score'])
    print("\nDataFrame after removing duplicates (based on Name and Score):")
    print(unique_df)

if __name__ == "__main__":
    main()
import pandas as pd

def clean_dataset(df, columns=None, drop_duplicates=True, drop_na=True):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns (list, optional): Specific columns to check for nulls/duplicates. 
                               If None, uses all columns.
    drop_duplicates (bool): Whether to drop duplicate rows.
    drop_na (bool): Whether to drop rows with null values.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if columns is None:
        columns = cleaned_df.columns
    
    if drop_na:
        cleaned_df = cleaned_df.dropna(subset=columns)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns, keep='first')
    
    return cleaned_df

def filter_by_value(df, column, value, keep=True):
    """
    Filter DataFrame rows based on column value.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to filter on.
    value: Value to filter by.
    keep (bool): If True, keep rows where column == value.
                 If False, keep rows where column != value.
    
    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    if keep:
        return df[df[column] == value].copy()
    else:
        return df[df[column] != value].copy()
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_column(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def clean_data(input_file, output_file):
    df = load_dataset(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    clean_data("raw_data.csv", "cleaned_data.csv")