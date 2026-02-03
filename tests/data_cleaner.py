
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_threshold=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame to clean
        numeric_columns: list of numeric columns to process
        outlier_threshold: IQR threshold for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
            cleaned_df[f"{col}_normalized"] = normalize_minmax(cleaned_df, col)
            cleaned_df[f"{col}_standardized"] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def calculate_statistics(df, column):
    """
    Calculate descriptive statistics for a column.
    
    Args:
        df: pandas DataFrame
        column: column name
    
    Returns:
        Dictionary with statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats_dict = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75),
        'skewness': df[column].skew(),
        'kurtosis': df[column].kurtosis()
    }
    
    return stats_dict

def detect_skewed_columns(df, skew_threshold=0.5):
    """
    Detect columns with significant skewness.
    
    Args:
        df: pandas DataFrame
        skew_threshold: absolute skewness threshold
    
    Returns:
        List of skewed column names
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewed_cols = []
    
    for col in numeric_cols:
        skewness = df[col].skew()
        if abs(skewness) > skew_threshold:
            skewed_cols.append((col, skewness))
    
    return sorted(skewed_cols, key=lambda x: abs(x[1]), reverse=True)
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_zscore(self, threshold=3):
        df_clean = self.df.copy()
        for col in self.numeric_columns:
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            df_clean = df_clean[(z_scores < threshold) | df_clean[col].isna()]
        self.df = df_clean
        return self
    
    def normalize_minmax(self):
        for col in self.numeric_columns:
            col_min = self.df[col].min()
            col_max = self.df[col].max()
            if col_max > col_min:
                self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
        return self
    
    def fill_missing_median(self):
        for col in self.numeric_columns:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def summary(self):
        print(f"Original shape: {self.df.shape}")
        print(f"Numeric columns: {len(self.numeric_columns)}")
        print(f"Missing values:")
        print(self.df[self.numeric_columns].isnull().sum())

def clean_dataset(df):
    cleaner = DataCleaner(df)
    cleaner.remove_outliers_zscore().fill_missing_median().normalize_minmax()
    return cleaner.get_cleaned_data()import pandas as pd

def clean_data(input_file, output_file):
    """
    Load data from a CSV file, remove duplicate rows,
    fill missing numeric values with the column mean,
    and save the cleaned data to a new CSV file.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        df_cleaned = df.drop_duplicates()
        print(f"After removing duplicates: {df_cleaned.shape}")
        
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    clean_data("raw_data.csv", "cleaned_data.csv")import pandas as pd
import argparse
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
        subset (list): List of column names to consider for duplicates
        keep (str): Which duplicates to keep: 'first', 'last', or False
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
        final_count = len(cleaned_df)
        
        duplicates_removed = initial_count - final_count
        
        if output_file:
            cleaned_df.to_csv(output_file, index=False)
            print(f"Processed {input_file}")
            print(f"Removed {duplicates_removed} duplicate rows")
            print(f"Saved cleaned data to {output_file}")
        else:
            print(f"Removed {duplicates_removed} duplicate rows")
            print(f"Original rows: {initial_count}, Cleaned rows: {final_count}")
        
        return cleaned_df
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Remove duplicate rows from CSV files')
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output CSV file path')
    parser.add_argument('-s', '--subset', nargs='+', help='Columns to consider for duplicates')
    parser.add_argument('-k', '--keep', choices=['first', 'last', 'none'], 
                       default='first', help="Which duplicates to keep")
    
    args = parser.parse_args()
    
    keep_value = False if args.keep == 'none' else args.keep
    
    remove_duplicates(
        input_file=args.input,
        output_file=args.output,
        subset=args.subset,
        keep=keep_value
    )

if __name__ == '__main__':
    main()