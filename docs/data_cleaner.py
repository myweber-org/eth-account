
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns, method='minmax'):
    df_norm = df.copy()
    for col in columns:
        if method == 'minmax':
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        elif method == 'zscore':
            df_norm[col] = stats.zscore(df_norm[col])
        elif method == 'log':
            df_norm[col] = np.log1p(df_norm[col])
    return df_norm

def clean_dataset(file_path, numeric_columns, outlier_removal=True, normalization='minmax'):
    df = pd.read_csv(file_path)
    
    if outlier_removal:
        df = remove_outliers_iqr(df, numeric_columns)
    
    if normalization:
        df = normalize_data(df, numeric_columns, method=normalization)
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset(
        'sample_data.csv',
        ['age', 'income', 'score'],
        outlier_removal=True,
        normalization='zscore'
    )
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(cleaned_df.head())
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for detecting outliers ('iqr', 'zscore')
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].dtype in [np.float64, np.int64]:
            handle_missing_values(df_clean, col, missing_strategy)
            handle_outliers(df_clean, col, outlier_method)
    
    return df_clean

def handle_missing_values(df, column, strategy='mean'):
    """Handle missing values in a specific column."""
    if df[column].isnull().sum() == 0:
        return
    
    if strategy == 'mean':
        fill_value = df[column].mean()
    elif strategy == 'median':
        fill_value = df[column].median()
    elif strategy == 'mode':
        fill_value = df[column].mode()[0]
    elif strategy == 'drop':
        df.dropna(subset=[column], inplace=True)
        return
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")
    
    df[column].fillna(fill_value, inplace=True)

def handle_outliers(df, column, method='iqr'):
    """Detect and cap outliers in a specific column."""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        z_scores = np.abs((df[column] - mean_val) / std_val)
        
        threshold = 3
        df[column] = np.where(z_scores > threshold, mean_val, df[column])
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"Dataset must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"

def get_dataset_summary(df):
    """Generate a summary of the dataset."""
    summary = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 60],
        'C': ['a', 'b', 'c', 'd', 'e', 'f']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nDataset summary:")
    print(get_dataset_summary(df))
    
    cleaned_df = clean_dataset(df, missing_strategy='mean', outlier_method='iqr')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    is_valid, message = validate_dataset(cleaned_df, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean'):
    """
    Load a CSV file, clean missing values, and return cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: File is empty.")
        return None

    print(f"Original shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")

    if fill_method == 'mean':
        df_cleaned = df.fillna(df.mean(numeric_only=True))
    elif fill_method == 'median':
        df_cleaned = df.fillna(df.median(numeric_only=True))
    elif fill_method == 'mode':
        df_cleaned = df.fillna(df.mode().iloc[0])
    elif fill_method == 'drop':
        df_cleaned = df.dropna()
    else:
        print("Warning: Unknown fill method. Using forward fill.")
        df_cleaned = df.fillna(method='ffill')

    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Remaining missing values: {df_cleaned.isnull().sum().sum()}")

    return df_cleaned

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a column using the IQR method.
    """
    if column not in df.columns:
        print(f"Error: Column '{column}' not found.")
        return None

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, np.nan, 8, 10],
        'C': [10, 20, 30, 40, 50]
    }
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_data.csv', index=False)

    cleaned_df = clean_csv_data('sample_data.csv', fill_method='mean')
    if cleaned_df is not None:
        print("\nCleaned DataFrame:")
        print(cleaned_df)

        outliers = detect_outliers_iqr(cleaned_df, 'C')
        if outliers is not None and not outliers.empty:
            print(f"\nOutliers in column 'C':")
            print(outliers)
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a dataset by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for handling outliers ('iqr', 'zscore', 'percentile')
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in df_clean.columns]
    
    # Handle missing values
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            if missing_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif missing_strategy == 'median':
                fill_value = df_clean[col].median()
            elif missing_strategy == 'mode':
                fill_value = df_clean[col].mode()[0]
            elif missing_strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown missing strategy: {missing_strategy}")
            
            df_clean[col] = df_clean[col].fillna(fill_value)
    
    # Handle outliers
    for col in numeric_cols:
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
        
        elif outlier_method == 'zscore':
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            z_scores = np.abs((df_clean[col] - mean) / std)
            df_clean = df_clean[z_scores < 3]
        
        elif outlier_method == 'percentile':
            lower_bound = df_clean[col].quantile(0.01)
            upper_bound = df_clean[col].quantile(0.99)
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
        
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, message)
    """
    if len(df) < min_rows:
        return False, f"Dataset must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if df.isnull().all().any():
        return False, "Some columns contain only null values"
    
    return True, "Dataset is valid"

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize numeric columns in the dataset.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to normalize, if None normalize all numeric columns
    method (str): Normalization method ('minmax', 'standard', 'robust')
    
    Returns:
    pd.DataFrame: Normalized DataFrame
    """
    df_norm = df.copy()
    
    if columns is None:
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in df_norm.columns]
    
    for col in numeric_cols:
        if method == 'minmax':
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        
        elif method == 'standard':
            mean = df_norm[col].mean()
            std = df_norm[col].std()
            if std > 0:
                df_norm[col] = (df_norm[col] - mean) / std
        
        elif method == 'robust':
            median = df_norm[col].median()
            iqr = df_norm[col].quantile(0.75) - df_norm[col].quantile(0.25)
            if iqr > 0:
                df_norm[col] = (df_norm[col] - median) / iqr
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return df_norm

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, 8, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, missing_strategy='mean', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid, message = validate_data(cleaned_df, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")
    
    normalized_df = normalize_data(cleaned_df, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized_df)