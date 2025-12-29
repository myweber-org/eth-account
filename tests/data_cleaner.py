
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using IQR method for specified columns.
    Returns cleaned DataFrame and outlier indices.
    """
    df_clean = df.copy()
    outlier_indices = []
    
    for col in columns:
        if col in df.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            outlier_indices.extend(outliers.index.tolist())
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    outlier_indices = list(set(outlier_indices))
    return df_clean, outlier_indices

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns cleaned DataFrame and outlier indices.
    """
    df_clean = df.copy()
    outlier_indices = []
    
    for col in columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            outliers = df_clean[z_scores > threshold]
            outlier_indices.extend(outliers.index.tolist())
            
            df_clean = df_clean[z_scores <= threshold]
    
    outlier_indices = list(set(outlier_indices))
    return df_clean, outlier_indices

def normalize_minmax(df, columns):
    """
    Normalize specified columns using Min-Max scaling.
    Returns DataFrame with normalized columns.
    """
    df_normalized = df.copy()
    
    for col in columns:
        if col in df.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            
            if max_val != min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
            else:
                df_normalized[col] = 0
    
    return df_normalized

def normalize_zscore(df, columns):
    """
    Normalize specified columns using Z-score standardization.
    Returns DataFrame with standardized columns.
    """
    df_standardized = df.copy()
    
    for col in columns:
        if col in df.columns:
            mean_val = df_standardized[col].mean()
            std_val = df_standardized[col].std()
            
            if std_val != 0:
                df_standardized[col] = (df_standardized[col] - mean_val) / std_val
            else:
                df_standardized[col] = 0
    
    return df_standardized

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing.
    """
    if outlier_method == 'iqr':
        df_clean, outliers = remove_outliers_iqr(df, numeric_columns)
    elif outlier_method == 'zscore':
        df_clean, outliers = remove_outliers_zscore(df, numeric_columns)
    else:
        df_clean, outliers = df.copy(), []
    
    if normalize_method == 'minmax':
        df_final = normalize_minmax(df_clean, numeric_columns)
    elif normalize_method == 'zscore':
        df_final = normalize_zscore(df_clean, numeric_columns)
    else:
        df_final = df_clean
    
    return df_final, outliers

def validate_data(df, numeric_columns):
    """
    Validate data by checking for missing values and data types.
    Returns validation report.
    """
    report = {
        'missing_values': df[numeric_columns].isnull().sum().to_dict(),
        'data_types': df[numeric_columns].dtypes.to_dict(),
        'basic_stats': df[numeric_columns].describe().to_dict()
    }
    
    return report

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'feature1': [1, 2, 3, 4, 5, 100],
        'feature2': [10, 20, 30, 40, 50, 200],
        'category': ['A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature1', 'feature2']
    
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50)
    
    validation_report = validate_data(df, numeric_cols)
    print("Validation Report:")
    print(f"Missing values: {validation_report['missing_values']}")
    
    cleaned_df, removed_outliers = clean_dataset(
        df, 
        numeric_cols, 
        outlier_method='iqr', 
        normalize_method='minmax'
    )
    
    print(f"\nRemoved {len(removed_outliers)} outliers at indices: {removed_outliers}")
    print("\nCleaned and Normalized DataFrame:")
    print(cleaned_df)