
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df = df.dropna()
    return df

def calculate_statistics(df, column):
    return {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'skewness': stats.skew(df[column].dropna()),
        'kurtosis': stats.kurtosis(df[column].dropna())
    }

if __name__ == "__main__":
    cleaned_data = clean_dataset('sample_data.csv', ['age', 'income', 'score'])
    print(cleaned_data.head())
    
    stats_result = calculate_statistics(cleaned_data, 'income')
    print(stats_result)
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(dataframe, columns):
    """
    Remove outliers using IQR method for specified columns.
    Returns cleaned dataframe and outlier indices.
    """
    cleaned_df = dataframe.copy()
    outlier_indices = []
    
    for col in columns:
        if col not in cleaned_df.columns:
            continue
            
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        col_outliers = cleaned_df[(cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)].index
        outlier_indices.extend(col_outliers)
        
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    
    return cleaned_df, list(set(outlier_indices))

def normalize_data(dataframe, columns, method='minmax'):
    """
    Normalize specified columns using different methods.
    Supported methods: 'minmax', 'zscore', 'robust'
    """
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            continue
            
        if method == 'minmax':
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            if col_max != col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
                
        elif method == 'zscore':
            col_mean = normalized_df[col].mean()
            col_std = normalized_df[col].std()
            if col_std != 0:
                normalized_df[col] = (normalized_df[col] - col_mean) / col_std
            else:
                normalized_df[col] = 0
                
        elif method == 'robust':
            col_median = normalized_df[col].median()
            col_iqr = stats.iqr(normalized_df[col])
            if col_iqr != 0:
                normalized_df[col] = (normalized_df[col] - col_median) / col_iqr
            else:
                normalized_df[col] = 0
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', fill_value=None):
    """
    Handle missing values in numeric columns.
    Strategies: 'mean', 'median', 'mode', 'constant', 'drop'
    """
    df_copy = dataframe.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df_copy.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if df_copy[col].isnull().any():
            if strategy == 'mean':
                fill_val = df_copy[col].mean()
            elif strategy == 'median':
                fill_val = df_copy[col].median()
            elif strategy == 'mode':
                fill_val = df_copy[col].mode()[0] if not df_copy[col].mode().empty else 0
            elif strategy == 'constant':
                fill_val = fill_value if fill_value is not None else 0
            else:
                fill_val = 0
                
            df_copy[col] = df_copy[col].fillna(fill_val)
    
    return df_copy

def clean_dataset(dataframe, numeric_columns=None, outlier_method='iqr', 
                  normalize_method='minmax', missing_strategy='mean'):
    """
    Main function to clean dataset with multiple steps.
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Original dataset shape: {dataframe.shape}")
    
    # Handle missing values
    cleaned_data = handle_missing_values(dataframe, strategy=missing_strategy)
    print(f"After handling missing values: {cleaned_data.shape}")
    
    # Remove outliers
    if outlier_method == 'iqr':
        cleaned_data, outliers = remove_outliers_iqr(cleaned_data, numeric_columns)
        print(f"Removed {len(outliers)} outlier rows")
        print(f"After outlier removal: {cleaned_data.shape}")
    
    # Normalize data
    normalized_data = normalize_data(cleaned_data, numeric_columns, method=normalize_method)
    
    return normalized_data

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Add some missing values
    sample_data.loc[np.random.choice(sample_data.index, 50), 'feature1'] = np.nan
    
    # Clean the dataset
    cleaned = clean_dataset(
        sample_data,
        numeric_columns=['feature1', 'feature2', 'feature3'],
        outlier_method='iqr',
        normalize_method='minmax',
        missing_strategy='mean'
    )
    
    print(f"Final cleaned dataset shape: {cleaned.shape}")
    print(f"Cleaned dataset statistics:\n{cleaned.describe()}")