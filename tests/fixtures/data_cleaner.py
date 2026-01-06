
import pandas as pd

def clean_dataframe(df, column_name, threshold, keep_above=True):
    """
    Filters a DataFrame based on a numeric column threshold.
    Removes rows where the column value is NaN.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    df_clean = df.dropna(subset=[column_name])

    if keep_above:
        filtered_df = df_clean[df_clean[column_name] > threshold]
    else:
        filtered_df = df_clean[df_clean[column_name] <= threshold]

    return filtered_df.reset_index(drop=True)
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str, optional): Path for cleaned output CSV
    missing_strategy (str): Strategy for handling missing values
                           ('mean', 'median', 'drop', 'zero')
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    df = df.drop_duplicates()
    print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True)
    elif missing_strategy == 'median':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
    elif missing_strategy == 'zero':
        df.fillna(0, inplace=True)
    elif missing_strategy == 'drop':
        df.dropna(inplace=True)
    
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} missing values remain after cleaning")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing columns: {missing_cols}")
    
    validation_results['summary']['rows'] = len(df)
    validation_results['summary']['columns'] = len(df.columns)
    validation_results['summary']['missing_values'] = df.isnull().sum().sum()
    validation_results['summary']['duplicates'] = df.duplicated().sum()
    
    return validation_results

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, np.nan, 15.2, np.nan, 20.1, 20.1],
        'category': ['A', 'B', 'A', 'C', 'B', 'B']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', 
                               'cleaned_sample.csv',
                               missing_strategy='mean')
    
    validation = validate_dataframe(cleaned_df, ['id', 'value', 'category'])
    print(f"Validation results: {validation}")
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
    cleaned_data = clean_dataset('raw_data.csv')
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print("Data cleaning completed. Saved to cleaned_data.csv")