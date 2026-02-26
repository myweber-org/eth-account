
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
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
    
    return filtered_df

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

def calculate_statistics(df):
    """
    Calculate basic statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Statistics summary
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    stats = {
        'mean': numeric_df.mean(),
        'median': numeric_df.median(),
        'std': numeric_df.std(),
        'min': numeric_df.min(),
        'max': numeric_df.max(),
        'count': numeric_df.count()
    }
    
    return pd.DataFrame(stats)
import pandas as pd
import numpy as np

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

def clean_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_minmax(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Original shape: {pd.read_csv(input_path).shape}")
    print(f"Cleaned shape: {df.shape}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column].apply(lambda x: 0.0)
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0.0)
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    if strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    elif strategy == 'mode':
        return data.fillna(data.mode().iloc[0])
    else:
        return data.fillna(0)

def clean_dataset(data, outlier_columns=None, normalize_columns=None, standardize_columns=None, missing_strategy='mean'):
    cleaned_data = data.copy()
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    if normalize_columns:
        for col in normalize_columns:
            if col in cleaned_data.columns:
                cleaned_data[col + '_normalized'] = normalize_minmax(cleaned_data, col)
    
    if standardize_columns:
        for col in standardize_columns:
            if col in cleaned_data.columns:
                cleaned_data[col + '_standardized'] = standardize_zscore(cleaned_data, col)
    
    return cleaned_data

def get_data_summary(data):
    summary = {
        'rows': len(data),
        'columns': len(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(data.select_dtypes(include=['object']).columns)
    }
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 100),
        'B': np.random.uniform(0, 1, 100),
        'C': np.random.randint(1, 50, 100)
    })
    
    sample_data.loc[10:15, 'A'] = np.nan
    sample_data.loc[20, 'B'] = 999
    
    print("Original data shape:", sample_data.shape)
    print("Data summary:", get_data_summary(sample_data))
    
    cleaned = clean_dataset(
        sample_data,
        outlier_columns=['A'],
        normalize_columns=['B'],
        standardize_columns=['C'],
        missing_strategy='mean'
    )
    
    print("\nCleaned data shape:", cleaned.shape)
    print("Cleaned data summary:", get_data_summary(cleaned))
    print("\nCleaned data columns:", list(cleaned.columns))
import pandas as pd

def clean_dataset(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values.
    rename_columns (bool): If True, rename columns to lowercase with underscores.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df.columns = (
            cleaned_df.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^a-z0-9_]', '', regex=True)
        )
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'User ID': [1, 2, None, 4],
        'First Name': ['Alice', 'Bob', 'Charlie', None],
        'Order Amount': [100.0, 150.0, 200.0, 250.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataset(cleaned, ['user_id', 'order_amount'])
    print(f"\nDataset validation: {is_valid}")import numpy as np
import pandas as pd
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

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
            cleaned_df = standardize_zscore(cleaned_df, col)
    return cleaned_df

def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True
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
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
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
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_standardized = self.df.copy()
        for col in columns:
            if col in df_standardized.columns and pd.api.types.is_numeric_dtype(df_standardized[col]):
                mean_val = df_standardized[col].mean()
                std_val = df_standardized[col].std()
                if std_val > 0:
                    df_standardized[col] = (df_standardized[col] - mean_val) / std_val
        
        self.df = df_standardized
        return self
    
    def handle_missing(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns and df_filled[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_filled[col].mean()
                elif strategy == 'median':
                    fill_value = df_filled[col].median()
                elif strategy == 'mode':
                    fill_value = df_filled[col].mode()[0]
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    continue
                
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self
    
    def get_dataframe(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'removed_rows': self.original_shape[0] - self.df.shape[0],
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(exclude=[np.number]).columns)
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'feature_b'] = 9999
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Sample data created with shape:", sample_df.shape)
    
    cleaner = DataCleaner(sample_df)
    print("\nInitial summary:", cleaner.get_summary())
    
    removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"\nRemoved {removed} outliers using IQR method")
    
    cleaner.handle_missing(strategy='mean')
    print("Handled missing values")
    
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    print("Applied min-max normalization")
    
    print("\nFinal summary:", cleaner.get_summary())
    print("\nFirst 5 rows of cleaned data:")
    print(cleaner.get_dataframe().head())import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Strategy to fill missing values. 
                        Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if fill_missing == 'drop':
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna()
        print(f"Removed {initial_rows - len(cleaned_df)} rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        for column in cleaned_df.select_dtypes(include=['number']).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                
                cleaned_df[column].fillna(fill_value, inplace=True)
                print(f"Filled missing values in column '{column}' with {fill_missing}: {fill_value:.2f}")
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    
    Returns:
    dict: Dictionary containing validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict(),
        'numeric_columns': list(df.select_dtypes(include=['number']).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None, 6],
        'B': [10, None, 30, 40, 50, 60],
        'C': ['a', 'b', 'b', 'c', 'd', 'e']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataframe(df))
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str): Method to fill missing values. Options are 'mean', 
                          'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        for column in cleaned_df.select_dtypes(include=['number']).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
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
    
    return True, "Dataset is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (fill with mean):")
    cleaned = clean_dataset(df, fill_missing='mean')
    print(cleaned)
    
    is_valid, message = validate_dataset(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (bool, str) indicating validation success and message.
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

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'x', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nValidation: {is_valid} - {message}")import numpy as np
import pandas as pd

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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.5)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_cleaning(df_before, df_after, column):
    stats_before = {
        'mean': df_before[column].mean(),
        'std': df_before[column].std(),
        'min': df_before[column].min(),
        'max': df_before[column].max()
    }
    stats_after = {
        'mean': df_after[column].mean(),
        'std': df_after[column].std(),
        'min': df_after[column].min(),
        'max': df_after[column].max()
    }
    return {'before': stats_before, 'after': stats_after}
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def normalize_column(dataframe, column, method='minmax'):
    if method == 'minmax':
        min_val = dataframe[column].min()
        max_val = dataframe[column].max()
        if max_val != min_val:
            dataframe[column] = (dataframe[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = dataframe[column].mean()
        std_val = dataframe[column].std()
        if std_val > 0:
            dataframe[column] = (dataframe[column] - mean_val) / std_val
    return dataframe

def clean_dataset(dataframe, numeric_columns):
    df_clean = dataframe.copy()
    for col in numeric_columns:
        df_clean = remove_outliers_iqr(df_clean, col)
        df_clean = normalize_column(df_clean, col, method='zscore')
    return df_clean

def validate_cleaned_data(dataframe, numeric_columns):
    validation_report = {}
    for col in numeric_columns:
        validation_report[col] = {
            'mean': round(dataframe[col].mean(), 4),
            'std': round(dataframe[col].std(), 4),
            'min': round(dataframe[col].min(), 4),
            'max': round(dataframe[col].max(), 4),
            'missing': dataframe[col].isnull().sum()
        }
    return validation_report
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                mask = (clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)
                clean_df = clean_df[mask]
        
        self.df = clean_df.reset_index(drop=True)
        return self
    
    def normalize_data(self, columns=None, method='minmax'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        
        for col in columns:
            if col in normalized_df.columns:
                if method == 'minmax':
                    min_val = normalized_df[col].min()
                    max_val = normalized_df[col].max()
                    if max_val != min_val:
                        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                
                elif method == 'zscore':
                    mean_val = normalized_df[col].mean()
                    std_val = normalized_df[col].std()
                    if std_val > 0:
                        normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        
        self.df = normalized_df
        return self
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_val = self.df[col].mean()
                elif strategy == 'median':
                    fill_val = self.df[col].median()
                elif strategy == 'mode':
                    fill_val = self.df[col].mode()[0]
                elif strategy == 'constant' and fill_value is not None:
                    fill_val = fill_value
                else:
                    continue
                
                self.df[col] = self.df[col].fillna(fill_val)
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def zscore_normalize(dataframe, column):
    """
    Normalize a column using z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    
    if std_val == 0:
        return dataframe[column] - mean_val
    
    normalized = (dataframe[column] - mean_val) / std_val
    return normalized

def minmax_normalize(dataframe, column, feature_range=(0, 1)):
    """
    Normalize a column using min-max normalization.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to normalize
        feature_range: Desired range of transformed data (default 0-1)
    
    Returns:
        Series with normalized values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if min_val == max_val:
        return pd.Series([feature_range[0]] * len(dataframe), index=dataframe.index)
    
    normalized = (dataframe[column] - min_val) / (max_val - min_val)
    normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return normalized

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect columns with significant skewness.
    
    Args:
        dataframe: pandas DataFrame
        threshold: Absolute skewness threshold (default 0.5)
    
    Returns:
        Dictionary with column names and their skewness values
    """
    skewed_columns = {}
    
    for column in dataframe.select_dtypes(include=[np.number]).columns:
        skewness = dataframe[column].skew()
        if abs(skewness) > threshold:
            skewed_columns[column] = skewness
    
    return skewed_columns

def log_transform(dataframe, column):
    """
    Apply log transformation to reduce skewness.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to transform
    
    Returns:
        Series with log-transformed values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if dataframe[column].min() <= 0:
        shifted = dataframe[column] - dataframe[column].min() + 1
        transformed = np.log(shifted)
    else:
        transformed = np.log(dataframe[column])
    
    return transformed

def clean_dataset(dataframe, numeric_columns=None, outlier_multiplier=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame to clean
        numeric_columns: List of numeric columns to process (default: all numeric)
        outlier_multiplier: IQR multiplier for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_multiplier)
    
    return cleaned_df

def create_cleaning_report(dataframe):
    """
    Generate a report of data quality issues.
    
    Args:
        dataframe: pandas DataFrame to analyze
    
    Returns:
        Dictionary with data quality metrics
    """
    report = {
        'total_rows': len(dataframe),
        'total_columns': len(dataframe.columns),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'numeric_columns': dataframe.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': dataframe.select_dtypes(include=['object']).columns.tolist(),
        'skewed_columns': detect_skewed_columns(dataframe),
        'column_dtypes': dataframe.dtypes.to_dict()
    }
    
    return report

if __name__ == "__main__":
    sample_data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.random.uniform(500, 1000, 50)
    
    print("Original dataset shape:", df.shape)
    
    cleaned_df = clean_dataset(df)
    print("Cleaned dataset shape:", cleaned_df.shape)
    
    report = create_cleaning_report(df)
    print("\nData quality report:")
    print(f"Total rows: {report['total_rows']}")
    print(f"Skewed columns: {len(report['skewed_columns'])}")
    
    normalized_feature = zscore_normalize(cleaned_df, 'feature_a')
    print(f"\nNormalized 'feature_a' statistics:")
    print(f"Mean: {normalized_feature.mean():.4f}")
    print(f"Std: {normalized_feature.std():.4f}")
import pandas as pd
import numpy as np

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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def main():
    sample_data = {
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    }
    df = pd.DataFrame(sample_data)
    print("Original dataset shape:", df.shape)
    
    numeric_cols = ['feature_a', 'feature_b']
    cleaned_df = clean_dataset(df, numeric_cols)
    print("Cleaned dataset shape:", cleaned_df.shape)
    print("Cleaned dataset head:")
    print(cleaned_df.head())

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers")
        return self
        
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                if method == 'zscore':
                    df_normalized[col] = stats.zscore(self.df[col])
                elif method == 'minmax':
                    df_normalized[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())
                elif method == 'robust':
                    median = self.df[col].median()
                    iqr = self.df[col].quantile(0.75) - self.df[col].quantile(0.25)
                    df_normalized[col] = (self.df[col] - median) / iqr
        
        self.df = df_normalized
        print(f"Normalized {len(columns)} columns using {method} method")
        return self
        
    def handle_missing_values(self, strategy='mean', fill_value=None):
        df_filled = self.df.copy()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_val = self.df[col].mean()
                elif strategy == 'median':
                    fill_val = self.df[col].median()
                elif strategy == 'mode':
                    fill_val = self.df[col].mode()[0]
                elif strategy == 'constant' and fill_value is not None:
                    fill_val = fill_value
                else:
                    continue
                    
                df_filled[col] = self.df[col].fillna(fill_val)
        
        self.df = df_filled
        print(f"Handled missing values using {strategy} strategy")
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def clean_dataset(df, outlier_removal=True, normalization=True, missing_values=True):
    cleaner = DataCleaner(df)
    
    if missing_values:
        cleaner.handle_missing_values()
    
    if outlier_removal:
        cleaner.remove_outliers_iqr()
    
    if normalization:
        cleaner.normalize_data()
    
    return cleaner.get_cleaned_data(), cleaner.get_summary()import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, numeric_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    for col in numeric_columns:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    
    return df
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
        Options: 'mean', 'median', 'mode', 'drop'.
    outlier_threshold (float): Number of standard deviations for outlier detection.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        for col in numeric_cols:
            cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
    elif missing_strategy == 'median':
        for col in numeric_cols:
            cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif missing_strategy == 'mode':
        for col in numeric_cols:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    elif missing_strategy == 'drop':
        cleaned_df.dropna(subset=numeric_cols, inplace=True)
    
    # Handle outliers using z-score method
    for col in numeric_cols:
        z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
        cleaned_df = cleaned_df[z_scores < outlier_threshold]
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if len(df) < min_rows:
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False
    
    return True

def calculate_statistics(df):
    """
    Calculate basic statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    dict: Dictionary containing statistics for each numeric column.
    """
    stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'count': df[col].count(),
            'missing': df[col].isnull().sum()
        }
    
    return stats

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df, missing_strategy='mean', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    stats = calculate_statistics(cleaned)
    print("\nStatistics:")
    for col, values in stats.items():
        print(f"{col}: {values}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data