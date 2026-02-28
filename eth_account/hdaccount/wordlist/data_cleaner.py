
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
import pandas as pd
import hashlib

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: {'first', 'last', False} which duplicates to keep
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def generate_hash(row):
    """
    Generate MD5 hash for a row to identify duplicates.
    
    Args:
        row: pandas Series representing a row
    
    Returns:
        MD5 hash string
    """
    row_string = str(row.to_dict()).encode('utf-8')
    return hashlib.md5(row_string).hexdigest()

def clean_dataset(input_file, output_file):
    """
    Main function to clean dataset by removing duplicates.
    
    Args:
        input_file: path to input CSV file
        output_file: path to save cleaned CSV file
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original dataset shape: {df.shape}")
        
        df_cleaned = remove_duplicates(df)
        print(f"Cleaned dataset shape: {df_cleaned.shape}")
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Removed {len(df) - len(df_cleaned)} duplicate rows")
        print(f"Cleaned data saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")

if __name__ == "__main__":
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    clean_dataset(input_path, output_path)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using IQR method
    """
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean.reset_index(drop=True)

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method
    """
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            df_clean = df_clean[(z_scores < threshold) | (df[col].isna())]
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns):
    """
    Normalize data using Min-Max scaling
    """
    df_norm = df.copy()
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
    return df_norm

def normalize_zscore(df, columns):
    """
    Normalize data using Z-score standardization
    """
    df_norm = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_norm[col] = (df[col] - mean_val) / std_val
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values with different strategies
    """
    df_filled = df.copy()
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if col in df.columns:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'zero':
                fill_value = 0
            else:
                fill_value = 0
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has fewer than {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True