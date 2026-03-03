import pandas as pd
import numpy as np

def detect_outliers_iqr(data, column):
    """
    Detect outliers using the Interquartile Range method.
    Returns a boolean mask where True indicates an outlier.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, column):
    """
    Remove outliers from a specified column using IQR method.
    Returns a cleaned DataFrame.
    """
    outlier_mask = detect_outliers_iqr(data, column)
    return data[~outlier_mask].copy()

def normalize_minmax(data, column):
    """
    Normalize a column using Min-Max scaling to range [0, 1].
    Returns a new Series with normalized values.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    return (data[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    """
    Main cleaning function: removes outliers and normalizes numeric columns.
    Returns a cleaned DataFrame and a dictionary of outlier counts per column.
    """
    cleaned_df = df.copy()
    outlier_info = {}
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            outliers = detect_outliers_iqr(cleaned_df, col)
            outlier_info[col] = outliers.sum()
            cleaned_df = remove_outliers(cleaned_df, col)
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
    
    return cleaned_df, outlier_info

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers from a column using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    """Remove outliers from a column using the Z-score method."""
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    """Normalize a column using Min-Max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    """Normalize a column using Z-score normalization."""
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(filepath, column, method='iqr', normalize='minmax'):
    """Main function to clean and normalize a dataset."""
    df = load_data(filepath)
    
    if method == 'iqr':
        df_clean = remove_outliers_iqr(df, column)
    elif method == 'zscore':
        df_clean = remove_outliers_zscore(df, column)
    else:
        df_clean = df.copy()
    
    if normalize == 'minmax':
        df_clean = normalize_minmax(df_clean, column)
    elif normalize == 'zscore':
        df_clean = normalize_zscore(df_clean, column)
    
    return df_clean

if __name__ == "__main__":
    cleaned_data = clean_dataset('sample_data.csv', 'value', method='iqr', normalize='minmax')
    print(cleaned_data.head())
    cleaned_data.to_csv('cleaned_data.csv', index=False)