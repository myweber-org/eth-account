
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_numeric_data(values, default=0):
    """
    Clean numeric data by converting strings to floats,
    handling None values, and replacing invalid entries with default.
    """
    cleaned = []
    for val in values:
        if val is None:
            cleaned.append(default)
        elif isinstance(val, str):
            try:
                cleaned.append(float(val))
            except ValueError:
                cleaned.append(default)
        else:
            cleaned.append(float(val))
    return cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    
    numeric_data = ["1.5", "invalid", None, 3.2, "4.7"]
    print("Numeric data:", numeric_data)
    print("Cleaned numeric:", clean_numeric_data(numeric_data))
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
    
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        print(f"Missing values found:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count} missing values")
        
        if missing_strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif missing_strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif missing_strategy == 'zero':
            df = df.fillna(0)
        elif missing_strategy == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    print(f"Final data shape: {df.shape}")
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate
    required_columns (list, optional): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            validation_results['warnings'].append(f"Column '{col}' contains negative values")
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 20.1],
        'category': ['A', 'B', 'A', None, 'C', 'C']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv', 'mean')
    
    validation = validate_dataframe(cleaned_df, ['id', 'value'])
    print(f"Validation results: {validation}")
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(df, numeric_columns):
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
        df = normalize_column(df, col)
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

def main():
    input_file = 'raw_data.csv'
    output_file = 'cleaned_data.csv'
    numeric_cols = ['age', 'income', 'score']
    
    raw_data = load_dataset(input_file)
    cleaned_data = clean_data(raw_data, numeric_cols)
    save_cleaned_data(cleaned_data, output_file)
    
    print(f"Data cleaning completed. Original shape: {raw_data.shape}")
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Cleaned data saved to: {output_file}")

if __name__ == "__main__":
    main()import numpy as np
import pandas as pd

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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True