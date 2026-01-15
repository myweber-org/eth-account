
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list): List of column names to check for duplicates. 
                             If None, checks all columns.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    original_shape = df.shape
    
    # Remove duplicates
    if columns_to_check is None:
        df_cleaned = df.drop_duplicates()
    else:
        df_cleaned = df.drop_duplicates(subset=columns_to_check)
    
    # Handle missing values
    if fill_missing == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
            elif fill_missing == 'median':
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in df_cleaned.columns:
            mode_value = df_cleaned[col].mode()
            if not mode_value.empty:
                df_cleaned[col].fillna(mode_value[0], inplace=True)
    
    # Report cleaning statistics
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    print(f"Original dataset shape: {original_shape}")
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Missing values handled using method: {fill_missing}")
    
    return df_cleaned

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has less than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, None]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'], fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid = validate_data(cleaned_df, required_columns=['id', 'name', 'age'], min_rows=3)
    print(f"\nData validation passed: {is_valid}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.copy()

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_clean = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_dfimport pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        missing_strategy (str): Strategy for handling missing values.
                               Options: 'mean', 'median', 'mode', 'drop', 'fill_zero'
        outlier_method (str): Method for detecting outliers.
                             Options: 'iqr', 'zscore', 'percentile'
        columns (list): Specific columns to clean. If None, clean all numeric columns.
    
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
            
        # Handle missing values
        if missing_strategy == 'mean':
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif missing_strategy == 'median':
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        elif missing_strategy == 'mode':
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        elif missing_strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif missing_strategy == 'fill_zero':
            df_clean[col].fillna(0, inplace=True)
        
        # Handle outliers
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
        
        elif outlier_method == 'zscore':
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            z_scores = np.abs((df_clean[col] - mean_val) / std_val)
            df_clean = df_clean[z_scores < 3]
        
        elif outlier_method == 'percentile':
            lower_bound = df_clean[col].quantile(0.01)
            upper_bound = df_clean[col].quantile(0.99)
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of columns that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to normalize. If None, normalize all numeric columns.
        method (str): Normalization method. Options: 'minmax', 'standard', 'robust'
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    df_norm = df.copy()
    
    if columns is None:
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df_norm.columns:
            continue
            
        if method == 'minmax':
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val != min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        
        elif method == 'standard':
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val != 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
        
        elif method == 'robust':
            median_val = df_norm[col].median()
            iqr_val = df_norm[col].quantile(0.75) - df_norm[col].quantile(0.25)
            if iqr_val != 0:
                df_norm[col] = (df_norm[col] - median_val) / iqr_val
    
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
    print(normalized_df)import pandas as pd
import numpy as np

def clean_dataset(df, columns=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Specific columns to clean, if None clean all columns
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if columns is None:
        columns = cleaned_df.columns.tolist()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
    
    for col in columns:
        if col not in cleaned_df.columns:
            continue
            
        missing_count = cleaned_df[col].isnull().sum()
        if missing_count > 0:
            print(f"Column '{col}' has {missing_count} missing values")
            
            if fill_missing == 'drop':
                cleaned_df = cleaned_df.dropna(subset=[col])
                print(f"Dropped rows with missing values in column '{col}'")
            elif fill_missing == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                fill_value = cleaned_df[col].mean()
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with mean: {fill_value:.2f}")
            elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                fill_value = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with median: {fill_value:.2f}")
            elif fill_missing == 'mode':
                fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None
                if fill_value is not None:
                    cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                    print(f"Filled missing values in column '{col}' with mode: {fill_value}")
            else:
                print(f"Warning: Could not fill missing values in column '{col}' with method '{fill_missing}'")
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a specific column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    method (str): Detection method ('iqr' or 'zscore')
    threshold (float): Threshold for outlier detection
    
    Returns:
    pd.Series: Boolean series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        z_scores = np.abs((df[column] - mean) / std)
        outliers = z_scores > threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    outlier_count = outliers.sum()
    if outlier_count > 0:
        print(f"Detected {outlier_count} outliers in column '{column}' using {method} method")
    
    return outliers

def validate_data_types(df, expected_types):
    """
    Validate that columns have expected data types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    expected_types (dict): Dictionary mapping column names to expected dtypes
    
    Returns:
    dict: Validation results
    """
    results = {}
    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            is_valid = actual_type == expected_type
            results[col] = {
                'expected': expected_type,
                'actual': actual_type,
                'valid': is_valid
            }
            if not is_valid:
                print(f"Warning: Column '{col}' has type {actual_type}, expected {expected_type}")
        else:
            results[col] = {
                'expected': expected_type,
                'actual': None,
                'valid': False,
                'error': 'Column not found'
            }
            print(f"Error: Column '{col}' not found in DataFrame")
    
    return results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, 20.3, np.nan, 40.1, 50.0, 50.0, 1000.0],
        'category': ['A', 'B', 'A', 'B', 'A', 'A', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    outliers = detect_outliers(cleaned, 'value')
    print(f"\nOutliers in 'value' column: {outliers.sum()}")
    
    expected_types = {
        'id': 'int64',
        'value': 'float64',
        'category': 'object'
    }
    
    validation = validate_data_types(cleaned, expected_types)
    print(f"\nData type validation: {validation}")
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
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        return self.df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
        
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_stats': {}
        }
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['numeric_stats'][col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max()
            }
        
        return summary
    
    def save_cleaned_data(self, filepath):
        self.df.to_csv(filepath, index=False)
        return filepath

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    data['feature_a'][np.random.choice(1000, 10)] = np.nan
    data['feature_b'][np.random.choice(1000, 5)] = 1000
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Initial shape:", cleaner.original_shape)
    print("Missing values:", cleaner.df.isnull().sum().sum())
    
    removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"Removed {removed} outliers")
    
    cleaner.fill_missing_median()
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    
    summary = cleaner.get_summary()
    print(f"Final shape: {summary['current_rows']} rows")
    print("Numeric stats:", summary['numeric_stats'].keys())
    
    cleaner.save_cleaned_data('cleaned_data.csv')
    print("Data saved to cleaned_data.csv")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: Boolean indicating whether to remove duplicate rows
        fill_missing: Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Removed rows with missing values")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                else:  # mode
                    fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Boolean indicating whether validation passed
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate a summary of the DataFrame including basic statistics.
    
    Args:
        df: pandas DataFrame to summarize
    
    Returns:
        Dictionary containing summary information
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
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