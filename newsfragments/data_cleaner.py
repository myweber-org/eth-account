
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df: pandas DataFrame to clean
        columns_to_check: list of columns to check for duplicates (default: all columns)
        drop_duplicates: boolean indicating whether to drop duplicate rows
        fill_missing: strategy to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    original_shape = df.shape
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Handle duplicates
    if drop_duplicates:
        if columns_to_check:
            cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
        else:
            cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_missing != 'drop':
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    else:
        cleaned_df = cleaned_df.dropna()
    
    # Log cleaning results
    print(f"Original dataset shape: {original_shape}")
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Rows removed: {original_shape[0] - cleaned_df.shape[0]}")
    print(f"Columns: {original_shape[1]}")
    
    return cleaned_df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate the cleaned dataset for basic data quality checks.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of columns that must be present
        numeric_columns: list of columns that should be numeric
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'has_duplicates': df.duplicated().any(),
        'has_missing_values': df.isnull().any().any(),
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
    
    # Check numeric columns
    if numeric_columns:
        non_numeric_cols = []
        for col in numeric_columns:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                non_numeric_cols.append(col)
        validation_results['non_numeric_columns'] = non_numeric_cols
    
    return validation_results

# Example usage (commented out for production)
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10, 20, np.nan, 30, 40, 50],
        'category': ['A', 'B', 'C', 'C', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns_to_check=['id'], fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    validation = validate_data(cleaned_df, required_columns=['id', 'value'], numeric_columns=['id', 'value'])
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, col)
        
        if normalize_method == 'minmax':
            cleaned_data[col] = normalize_minmax(cleaned_data, col)
        elif normalize_method == 'zscore':
            cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate statistical summary of the dataset.
    """
    summary = {
        'rows': len(data),
        'columns': len(data.columns),
        'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(data.select_dtypes(include=['object']).columns),
        'missing_values': data.isnull().sum().to_dict(),
        'data_types': data.dtypes.to_dict()
    }
    
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['X', 'Y', 'Z'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("\nData summary:")
    summary = get_data_summary(sample_data)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'], outlier_method='iqr', normalize_method='minmax')
    print("\nCleaned data shape:", cleaned.shape)
    print("\nCleaned data summary:")
    cleaned_summary = get_data_summary(cleaned)
    for key, value in cleaned_summary.items():
        print(f"{key}: {value}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data (list or np.array): Input data
        column (int): Index of column to process (for 2D arrays)
    
    Returns:
        np.array: Data with outliers removed
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if data.ndim == 1:
        column_data = data
    else:
        column_data = data[:, column]
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if data.ndim == 1:
        filtered_data = column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]
    else:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        filtered_data = data[mask]
    
    return filtered_data

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data (np.array): Input data
    
    Returns:
        dict: Dictionary containing statistics
    """
    if len(data) == 0:
        return {
            'mean': 0,
            'median': 0,
            'std': 0,
            'min': 0,
            'max': 0
        }
    
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }

def clean_dataset(data, column=0):
    """
    Main function to clean dataset by removing outliers.
    
    Args:
        data (list or np.array): Input data
        column (int): Column index for 2D arrays
    
    Returns:
        tuple: (cleaned_data, stats_before, stats_after)
    """
    if isinstance(data, list):
        data = np.array(data)
    
    stats_before = calculate_statistics(data if data.ndim == 1 else data[:, column])
    
    cleaned_data = remove_outliers_iqr(data, column)
    
    stats_after = calculate_statistics(cleaned_data if data.ndim == 1 else cleaned_data[:, column])
    
    return cleaned_data, stats_before, stats_after

if __name__ == "__main__":
    test_data = np.random.normal(100, 15, 1000)
    test_data[50] = 500
    test_data[150] = -200
    
    cleaned, before, after = clean_dataset(test_data)
    
    print(f"Original data points: {len(test_data)}")
    print(f"Cleaned data points: {len(cleaned)}")
    print(f"Outliers removed: {len(test_data) - len(cleaned)}")
    print(f"\nStatistics before cleaning: {before}")
    print(f"Statistics after cleaning: {after}")
import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    strategy (str): Strategy for missing value imputation ('mean', 'median', 'mode', 'drop').
    outlier_threshold (float): Number of standard deviations to consider as outlier.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif strategy in ['mean', 'median']:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if strategy == 'mean':
                fill_value = cleaned_df[column].mean()
            else:
                fill_value = cleaned_df[column].median()
            cleaned_df[column].fillna(fill_value, inplace=True)
    elif strategy == 'mode':
        for column in cleaned_df.columns:
            fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else None
            if fill_value is not None:
                cleaned_df[column].fillna(fill_value, inplace=True)
    
    # Handle outliers for numerical columns
    numerical_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for column in numerical_cols:
        mean = cleaned_df[column].mean()
        std = cleaned_df[column].std()
        if std > 0:  # Avoid division by zero
            z_scores = np.abs((cleaned_df[column] - mean) / std)
            cleaned_df = cleaned_df[z_scores < outlier_threshold]
    
    # Reset index after outlier removal
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],  # Contains outlier (100) and missing value
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned = clean_dataset(df, strategy='mean', outlier_threshold=2)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B', 'C'], min_rows=2)
    print(f"Validation result: {is_valid}")
    print(f"Validation message: {message}")import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for identifying duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    rows_removed = len(df) - len(cleaned_df)
    
    if rows_removed > 0:
        print(f"Removed {rows_removed} duplicate rows")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
    
    Returns:
        Boolean indicating if validation passed
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate dtype and handling errors.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def detect_outliers_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        self.df = self.df[(z_scores < threshold) | (self.df[column].isna())]
        return self
    
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            self.df[column] = (self.df[column] - mean_val) / std_val
        return self
    
    def fill_missing(self, column, strategy='mean'):
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = strategy
            
        self.df[column].fillna(fill_value, inplace=True)
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]