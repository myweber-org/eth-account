
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def zscore_normalization(df, column):
    """Normalize column using Z-score method."""
    df[column + '_normalized'] = stats.zscore(df[column])
    return df

def minmax_normalization(df, column):
    """Normalize column using Min-Max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_scaled'] = (df[column] - min_val) / (max_val - min_val)
    return df

def handle_missing_values(df, strategy='mean'):
    """Handle missing values with specified strategy."""
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.dropna()

def clean_dataset(input_file, output_file):
    """Main data cleaning pipeline."""
    df = load_dataset(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        df = remove_outliers_iqr(df, column)
    
    df = handle_missing_values(df, strategy='median')
    
    for column in numeric_columns:
        df = zscore_normalization(df, column)
        df = minmax_normalization(df, column)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')