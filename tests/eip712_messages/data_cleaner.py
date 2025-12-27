
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
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

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

def example_usage():
    """
    Demonstrate the usage of data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.uniform(30, 80, 100)
    }
    df = pd.DataFrame(data)

    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary statistics for temperature:")
    print(calculate_summary_statistics(df, 'temperature'))

    cleaned_df = remove_outliers_iqr(df, 'temperature')
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned summary statistics for temperature:")
    print(calculate_summary_statistics(cleaned_df, 'temperature'))

    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def z_score_normalize(data, column):
    """
    Normalize data using z-score method
    """
    mean = data[column].mean()
    std = data[column].std()
    data[column + '_zscore'] = (data[column] - mean) / std
    return data

def clean_dataset(df, numeric_columns):
    """
    Main cleaning function for numeric columns
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
            cleaned_df = z_score_normalize(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that required columns exist and have no null values
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Columns with null values: {null_counts[null_counts > 0].to_dict()}")
    
    return True
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(file_path: str, 
                   missing_strategy: str = 'drop',
                   fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    file_path: Path to the CSV file
    missing_strategy: Strategy for handling missing values ('drop', 'fill', 'interpolate')
    fill_value: Value to use when missing_strategy is 'fill'
    
    Returns:
    Cleaned pandas DataFrame
    """
    
    try:
        df = pd.read_csv(file_path)
        
        if missing_strategy == 'drop':
            df_clean = df.dropna()
        elif missing_strategy == 'fill':
            if fill_value is not None:
                df_clean = df.fillna(fill_value)
            else:
                df_clean = df.fillna(df.mean(numeric_only=True))
        elif missing_strategy == 'interpolate':
            df_clean = df.interpolate(method='linear')
        else:
            raise ValueError("Invalid missing_strategy. Use 'drop', 'fill', or 'interpolate'")
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_clean[numeric_cols] = df_clean[numeric_cols].round(4)
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print("Error: The file is empty")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return pd.DataFrame()

def validate_dataframe(df: pd.DataFrame, 
                      required_columns: list = None) -> bool:
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df: DataFrame to validate
    required_columns: List of columns that must be present
    
    Returns:
    Boolean indicating if DataFrame is valid
    """
    if df.empty:
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.isnull().sum().sum() > 0:
        print("Warning: DataFrame contains missing values")
    
    return True

def save_cleaned_data(df: pd.DataFrame, 
                     output_path: str, 
                     index: bool = False) -> bool:
    """
    Save cleaned DataFrame to CSV file.
    
    Parameters:
    df: DataFrame to save
    output_path: Path for output file
    index: Whether to save index
    
    Returns:
    Boolean indicating success
    """
    try:
        df.to_csv(output_path, index=index)
        print(f"Cleaned data saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return False

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [1.1, 2.2, 3.3, np.nan, 5.5],
        'C': ['x', 'y', 'z', 'w', 'v']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned = clean_csv_data('sample_data.csv', missing_strategy='fill')
    
    if validate_dataframe(cleaned):
        save_cleaned_data(cleaned, 'cleaned_sample_data.csv')
        print("Data cleaning completed successfully")
    else:
        print("Data validation failed")