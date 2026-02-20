
import pandas as pd
import re

def clean_dataframe(df, text_column):
    """
    Clean a DataFrame by removing duplicate rows and standardizing text in a specified column.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Standardize text: lowercase and remove extra whitespace
    if text_column in df_cleaned.columns:
        df_cleaned[text_column] = df_cleaned[text_column].apply(
            lambda x: re.sub(r'\s+', ' ', str(x).strip().lower())
        )
    
    return df_cleaned

def filter_by_keyword(df, text_column, keyword):
    """
    Filter rows where the specified text column contains a given keyword.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    filtered_df = df[df[text_column].str.contains(keyword, case=False, na=False)]
    return filtered_df.reset_index(drop=True)
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if not np.issubdtype(normalized_df[col].dtype, np.number):
            raise TypeError(f"Column '{col}' must be numeric for normalization")
        
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        
        if col_max == col_min:
            normalized_df[col] = 0.5
        else:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
    
    return normalized_df

def standardize_zscore(dataframe, columns=None):
    """
    Standardize specified columns using z-score normalization.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to standardize. If None, standardize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    standardized_df = dataframe.copy()
    
    for col in columns:
        if col not in standardized_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if not np.issubdtype(standardized_df[col].dtype, np.number):
            raise TypeError(f"Column '{col}' must be numeric for standardization")
        
        col_mean = standardized_df[col].mean()
        col_std = standardized_df[col].std()
        
        if col_std == 0:
            standardized_df[col] = 0
        else:
            standardized_df[col] = (standardized_df[col] - col_mean) / col_std
    
    return standardized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of column names to process. If None, process all columns.
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns
    
    processed_df = dataframe.copy()
    
    for col in columns:
        if col not in processed_df.columns:
            continue
        
        if processed_df[col].isnull().sum() == 0:
            continue
        
        if strategy == 'drop':
            processed_df = processed_df.dropna(subset=[col])
        elif strategy == 'mean':
            if np.issubdtype(processed_df[col].dtype, np.number):
                processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
        elif strategy == 'median':
            if np.issubdtype(processed_df[col].dtype, np.number):
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        elif strategy == 'mode':
            processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return processed_df

def create_data_summary(dataframe):
    """
    Create a summary statistics DataFrame.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary statistics DataFrame
    """
    summary = pd.DataFrame({
        'column': dataframe.columns,
        'dtype': dataframe.dtypes.values,
        'non_null_count': dataframe.count().values,
        'null_count': dataframe.isnull().sum().values,
        'null_percentage': (dataframe.isnull().sum() / len(dataframe) * 100).values,
        'unique_count': dataframe.nunique().values
    })
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = dataframe[numeric_cols].describe().T
        summary = summary.merge(numeric_stats, left_on='column', right_index=True, how='left')
    
    return summary.reset_index(drop=True)
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns, method='minmax'):
    df_norm = df.copy()
    for col in columns:
        if col in df.columns:
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df_norm[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df_norm[col] = 0
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df_norm[col] = (df[col] - mean_val) / std_val
                else:
                    df_norm[col] = 0
    return df_norm

def clean_dataset(df, numeric_columns):
    df_no_outliers = remove_outliers_iqr(df, numeric_columns)
    df_normalized = normalize_data(df_no_outliers, numeric_columns, method='zscore')
    df_normalized = df_normalized.dropna()
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'feature1': [10, 12, 12, 14, 12, 11, 100, 12, 13, 12],
        'feature2': [1.2, 1.3, 1.1, 1.4, 1.2, 1.3, 5.0, 1.2, 1.1, 1.3],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature1', 'feature2']
    cleaned_df = clean_dataset(df, numeric_cols)
    print("Original dataset shape:", df.shape)
    print("Cleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned dataset:")
    print(cleaned_df)
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        return self
    
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                if method == 'zscore':
                    df_normalized[col] = stats.zscore(df_normalized[col])
                elif method == 'minmax':
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                elif method == 'robust':
                    median = df_normalized[col].median()
                    iqr = df_normalized[col].quantile(0.75) - df_normalized[col].quantile(0.25)
                    df_normalized[col] = (df_normalized[col] - median) / iqr
        
        self.df = df_normalized
        return self
    
    def handle_missing_values(self, columns=None, strategy='mean'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns and df_filled[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_filled[col].mean()
                elif strategy == 'median':
                    fill_value = df_filled[col].median()
                elif strategy == 'mode':
                    fill_value = df_filled[col].mode()[0]
                elif strategy == 'drop':
                    df_filled = df_filled.dropna(subset=[col])
                    continue
                else:
                    fill_value = strategy
                
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            original_count = len(df)
            df = remove_outliers_iqr(df, column)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column: {column}")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved to: {output_file}")
        print(f"Final dataset shape: {df.shape}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing values in a DataFrame using specified strategy.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values. 
                       Options: 'mean', 'median', 'mode', 'drop', 'fill_zero'
        columns (list): List of columns to apply cleaning. If None, applies to all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df_clean[col].mean()
        elif strategy == 'median':
            fill_value = df_clean[col].median()
        elif strategy == 'mode':
            fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
        elif strategy == 'fill_zero':
            fill_value = 0
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
        method (str): Normalization method. Options: 'minmax', 'zscore'
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df_normalized[column].min()
        max_val = df_normalized[column].max()
        if max_val != min_val:
            df_normalized[column] = (df_normalized[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_normalized[column].mean()
        std_val = df_normalized[column].std()
        if std_val != 0:
            df_normalized[column] = (df_normalized[column] - mean_val) / std_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df_normalized

def load_and_clean_csv(filepath, **kwargs):
    """
    Load CSV file and apply cleaning operations.
    
    Args:
        filepath (str): Path to CSV file
        **kwargs: Arguments to pass to clean_missing_data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        df_clean = clean_missing_data(df, **kwargs)
        return df_clean
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading file: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': [1, 2, 3, 4, 5]
    }
    
    df_sample = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df_sample)
    
    df_cleaned = clean_missing_data(df_sample, strategy='mean')
    print("\nCleaned DataFrame (mean imputation):")
    print(df_cleaned)