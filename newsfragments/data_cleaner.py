
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, fill_strategy='mean'):
    """
    Load a CSV file, clean missing values, and optionally save the cleaned data.
    
    Parameters:
    file_path (str): Path to the input CSV file.
    output_path (str, optional): Path to save the cleaned CSV. If None, no file is saved.
    fill_strategy (str): Strategy for filling missing values. Options: 'mean', 'median', 'mode', 'zero'.
    
    Returns:
    pandas.DataFrame: The cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

    if fill_strategy == 'mean':
        for col in numeric_columns:
            df[col].fillna(df[col].mean(), inplace=True)
    elif fill_strategy == 'median':
        for col in numeric_columns:
            df[col].fillna(df[col].median(), inplace=True)
    elif fill_strategy == 'mode':
        for col in numeric_columns:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
    elif fill_strategy == 'zero':
        df.fillna(0, inplace=True)
    else:
        raise ValueError("Invalid fill_strategy. Choose from 'mean', 'median', 'mode', 'zero'.")

    for col in non_numeric_columns:
        df[col].fillna('Unknown', inplace=True)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")

    return df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a specific column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    column (str): The column name to process.
    multiplier (float): The multiplier for IQR (default 1.5).
    
    Returns:
    pandas.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, np.nan, 30, 40, 50, 60],
        'C': ['X', 'Y', np.nan, 'Z', 'X', 'Y']
    }
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv', fill_strategy='median')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    df_no_outliers = remove_outliers_iqr(cleaned_df, 'A')
    print("\nDataFrame after removing outliers from column 'A':")
    print(df_no_outliers)