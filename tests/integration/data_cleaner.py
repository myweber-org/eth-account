import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column using min-max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column]
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize column using z-score normalization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_threshold=1.5, normalization_method='standardize'):
    """
    Clean dataset by removing outliers and applying normalization.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if normalization_method == 'minmax':
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
            elif normalization_method == 'standardize':
                cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, allow_nan=False):
    """
    Validate dataframe structure and content.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan:
        nan_columns = df.columns[df.isnull().any()].tolist()
        if nan_columns:
            raise ValueError(f"Columns with NaN values: {nan_columns}")
    
    return True

def sample_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    sample_data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[10:15, 'feature_a'] = 500
    df.loc[20:25, 'feature_b'] = 1000
    
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    
    try:
        validate_data(df, numeric_cols)
        cleaned_df = clean_dataset(df, numeric_cols, outlier_threshold=1.5, normalization_method='standardize')
        print(f"Original shape: {df.shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")
        print(f"Cleaned data summary:\n{cleaned_df.describe()}")
        return cleaned_df
    except ValueError as e:
        print(f"Validation error: {e}")
        return None

if __name__ == "__main__":
    cleaned_data = sample_usage()import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            data[col].fillna(data[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            data[col].fillna(data[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)
    elif strategy == 'drop':
        data.dropna(subset=numeric_cols, inplace=True)
    
    return data

def clean_dataset(data, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    for col in numeric_cols:
        if normalize_method == 'minmax':
            cleaned_data = normalize_minmax(cleaned_data, col)
        elif normalize_method == 'zscore':
            cleaned_data = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data):
    """
    Validate cleaned data
    """
    validation_report = {}
    
    validation_report['total_rows'] = len(data)
    validation_report['total_columns'] = len(data.columns)
    validation_report['missing_values'] = data.isnull().sum().sum()
    validation_report['duplicate_rows'] = data.duplicated().sum()
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        validation_report[f'{col}_mean'] = data[col].mean()
        validation_report[f'{col}_std'] = data[col].std()
        validation_report[f'{col}_min'] = data[col].min()
        validation_report[f'{col}_max'] = data[col].max()
    
    return validation_report

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000)
    })
    
    sample_data.loc[np.random.choice(sample_data.index, 50), 'feature1'] = np.nan
    sample_data.loc[np.random.choice(sample_data.index, 30), 'feature2'] = np.nan
    
    cleaned = clean_dataset(sample_data, outlier_method='iqr', normalize_method='minmax')
    report = validate_data(cleaned)
    
    print("Data cleaning completed successfully")
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Missing values in cleaned data: {report['missing_values']}")