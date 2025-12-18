import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val - min_val == 0:
        return data[column]
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_factor=1.5, normalization_method='minmax'):
    """
    Clean dataset by removing outliers and applying normalization.
    """
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            if normalization_method == 'minmax':
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
            elif normalization_method == 'zscore':
                cleaned_df[col] = standardize_zscore(cleaned_df, col)
    return cleaned_df

def main():
    sample_data = {
        'A': np.random.normal(100, 15, 100),
        'B': np.random.exponential(50, 100),
        'C': np.random.uniform(0, 1, 100)
    }
    df = pd.DataFrame(sample_data)
    print("Original dataset shape:", df.shape)
    print("Original statistics:")
    print(df.describe())
    
    cleaned_df = clean_dataset(df, ['A', 'B', 'C'], outlier_factor=1.5, normalization_method='minmax')
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Cleaned statistics:")
    print(cleaned_df.describe())

if __name__ == "__main__":
    main()