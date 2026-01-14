
import pandas as pd
import numpy as np
import sys

def clean_csv(input_path, output_path, strategy='mean'):
    """
    Clean a CSV file by handling missing values.
    
    Parameters:
    input_path (str): Path to the input CSV file.
    output_path (str): Path to save the cleaned CSV file.
    strategy (str): Strategy for handling missing values.
                    Options: 'mean', 'median', 'mode', 'drop'.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Original data shape: {df.shape}")
        
        if strategy == 'drop':
            df_cleaned = df.dropna()
        elif strategy in ['mean', 'median']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if strategy == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            else:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            df_cleaned = df
        elif strategy == 'mode':
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
                else:
                    df[col] = df[col].fillna(df[col].mean())
            df_cleaned = df
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data shape: {df_cleaned.shape}")
        print(f"Cleaned data saved to: {output_path}")
        
        missing_report = df_cleaned.isnull().sum()
        if missing_report.sum() > 0:
            print("Warning: Some missing values remain:")
            print(missing_report[missing_report > 0])
        else:
            print("No missing values remaining.")
            
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python data_cleaner.py <input_file> <output_file> [strategy]")
        print("Strategies: mean, median, mode, drop (default: mean)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    strategy = sys.argv[3] if len(sys.argv) > 3 else 'mean'
    
    clean_csv(input_file, output_file, strategy)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from specified column using IQR method.
    Returns filtered dataframe and outlier indices.
    """
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    mask = (dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)
    outliers = dataframe[~mask].index.tolist()
    
    return dataframe[mask].copy(), outliers

def normalize_minmax(dataframe, columns=None):
    """
    Apply min-max normalization to specified columns.
    If columns is None, normalize all numeric columns.
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    result = dataframe.copy()
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            if col_max > col_min:
                result[col] = (dataframe[col] - col_min) / (col_max - col_min)
    
    return result

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Identify columns with significant skewness.
    Returns dictionary with column names and skewness values.
    """
    skewed_cols = {}
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        skewness = stats.skew(dataframe[col].dropna())
        if abs(skewness) > threshold:
            skewed_cols[col] = skewness
    
    return skewed_cols

def clean_dataset(dataframe, drop_na_threshold=0.3):
    """
    Perform comprehensive data cleaning:
    1. Remove columns with too many missing values
    2. Remove duplicate rows
    3. Reset index
    """
    cleaned_df = dataframe.copy()
    
    # Remove columns with excessive missing values
    missing_ratios = cleaned_df.isnull().sum() / len(cleaned_df)
    cols_to_drop = missing_ratios[missing_ratios > drop_na_threshold].index.tolist()
    cleaned_df = cleaned_df.drop(columns=cols_to_drop)
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df, cols_to_drop

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    Returns boolean and error message if validation fails.
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Validation passed"