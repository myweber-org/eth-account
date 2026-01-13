import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_na_threshold=0.5):
    """
    Clean a pandas DataFrame by removing duplicates, handling missing values,
    and optionally renaming columns.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    
    # Rename columns if mapping is provided
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    # Calculate missing value percentage for each column
    missing_percent = df_clean.isnull().sum() / len(df_clean)
    
    # Drop columns with missing values above threshold
    columns_to_drop = missing_percent[missing_percent > drop_na_threshold].index
    df_clean = df_clean.drop(columns=columns_to_drop)
    
    # Fill remaining missing values with appropriate defaults
    for column in df_clean.columns:
        if df_clean[column].dtype == 'object':
            # For categorical columns, fill with mode
            if not df_clean[column].mode().empty:
                df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
            else:
                df_clean[column] = df_clean[column].fillna('Unknown')
        else:
            # For numerical columns, fill with median
            df_clean[column] = df_clean[column].fillna(df_clean[column].median())
    
    # Remove outliers using IQR method for numerical columns
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing rows
        df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
    
    # Standardize column names: lowercase with underscores
    df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    # Print cleaning summary
    print(f"Cleaning Summary:")
    print(f"- Original rows: {initial_rows}")
    print(f"- Duplicates removed: {duplicates_removed}")
    print(f"- Columns dropped due to high missing values: {list(columns_to_drop)}")
    print(f"- Final dataset shape: {df_clean.shape}")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic quality requirements.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for completely empty columns
    empty_columns = df.columns[df.isnull().all()]
    if len(empty_columns) > 0:
        print(f"Warning: Found completely empty columns: {list(empty_columns)}")
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'Customer ID': [1, 2, 2, 3, 4, 5, None],
        'Name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank', 'Grace'],
        'Age': [25, 30, 30, 35, 28, None, 40],
        'Salary': [50000, 60000, 60000, 75000, 55000, 48000, 90000],
        'Empty Column': [None, None, None, None, None, None, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    column_mapping = {'Customer ID': 'customer_id', 'Name': 'name', 'Age': 'age', 'Salary': 'salary'}
    cleaned_df = clean_dataset(df, column_mapping=column_mapping, drop_na_threshold=0.6)
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    try:
        validate_dataframe(cleaned_df, required_columns=['customer_id', 'name', 'age', 'salary'])
        print("\nData validation passed!")
    except ValueError as e:
        print(f"\nData validation failed: {e}")
import pandas as pd
import numpy as np
from scipy import stats

def normalize_column(data, column_name, method='minmax'):
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in data")
    
    column_data = data[column_name].values
    
    if method == 'minmax':
        min_val = np.min(column_data)
        max_val = np.max(column_data)
        if max_val == min_val:
            return data
        normalized = (column_data - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = np.mean(column_data)
        std_val = np.std(column_data)
        if std_val == 0:
            return data
        normalized = (column_data - mean_val) / std_val
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    data[column_name] = normalized
    return data

def remove_outliers_iqr(data, column_name):
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in data")
    
    column_data = data[column_name].values
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column_name, threshold=3):
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in data")
    
    column_data = data[column_name].values
    z_scores = np.abs(stats.zscore(column_data))
    
    filtered_data = data[z_scores < threshold]
    return filtered_data

def clean_dataset(data, numeric_columns=None, normalization_method='minmax', outlier_method='iqr'):
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = normalize_column(cleaned_data, column, normalization_method)
            
            if outlier_method == 'iqr':
                cleaned_data = remove_outliers_iqr(cleaned_data, column)
            elif outlier_method == 'zscore':
                cleaned_data = remove_outliers_zscore(cleaned_data, column)
            else:
                raise ValueError("Outlier method must be 'iqr' or 'zscore'")
    
    return cleaned_data

def get_data_summary(data):
    summary = {
        'original_rows': len(data),
        'original_columns': len(data.columns),
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum()
    }
    return summary
import numpy as np
import pandas as pd

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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def process_numerical_data(df, columns):
    """
    Process multiple numerical columns by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Processed DataFrame
    """
    result_df = df.copy()
    
    for col in columns:
        if col in result_df.columns and pd.api.types.is_numeric_dtype(result_df[col]):
            result_df = remove_outliers_iqr(result_df, col)
    
    return result_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original shape: {df.shape}")
    
    # Add some outliers
    df.loc[1000] = [500, 1000, 300]
    df.loc[1001] = [-100, -50, -10]
    
    processed_df = process_numerical_data(df, ['A', 'B', 'C'])
    print(f"Processed shape: {processed_df.shape}")
    
    for col in ['A', 'B', 'C']:
        stats = calculate_summary_statistics(processed_df, col)
        print(f"\nStatistics for {col}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

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
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 200],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary statistics before cleaning:")
    for col in df.columns:
        stats = calculate_summary_statistics(df, col)
        print(f"\n{col}: {stats}")
    
    cleaned_df = clean_numeric_data(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nSummary statistics after cleaning:")
    for col in cleaned_df.columns:
        stats = calculate_summary_statistics(cleaned_df, col)
        print(f"\n{col}: {stats}")
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                z_scores = np.abs(stats.zscore(self.df[col].fillna(self.df[col].mean())))
                df_clean = df_clean[z_scores < threshold]
        self.df = df_clean
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                col_min = self.df[col].min()
                col_max = self.df[col].max()
                if col_max > col_min:
                    self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
        return self
        
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def reset_to_original(self):
        self.df = self.original_df.copy()
        return self
        
    def get_summary(self):
        summary = {
            'original_rows': len(self.original_df),
            'cleaned_rows': len(self.df),
            'removed_rows': len(self.original_df) - len(self.df),
            'original_columns': list(self.original_df.columns),
            'cleaned_columns': list(self.df.columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    data['feature1'][np.random.choice(100, 5)] = np.nan
    data['feature1'][10:12] = [500, 600]
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    print("Original shape:", df.shape)
    cleaner.fill_missing_mean().remove_outliers_zscore().normalize_minmax()
    cleaned_df = cleaner.get_cleaned_data()
    print("Cleaned shape:", cleaned_df.shape)
    print("Summary:", cleaner.get_summary())
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print("Data cleaning completed successfully.")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process. If None, process all numeric columns.
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def zscore_normalize(df, columns=None):
    """
    Normalize data using Z-score normalization.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def minmax_normalize(df, columns=None, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    feature_range (tuple): Desired range of transformed data (default 0-1)
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            col_range = col_max - col_min
            
            if col_range > 0:
                df_normalized[col] = ((df[col] - col_min) / col_range) * (max_val - min_val) + min_val
    
    return df_normalized

def detect_missing_patterns(df, threshold=0.3):
    """
    Detect columns with high percentage of missing values.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    threshold (float): Threshold for missing value percentage (default 0.3)
    
    Returns:
    list: Columns with missing values above threshold
    """
    missing_percent = df.isnull().sum() / len(df)
    high_missing_cols = missing_percent[missing_percent > threshold].index.tolist()
    return high_missing_cols

def clean_dataset(df, outlier_columns=None, normalize_method='zscore', missing_threshold=0.3):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    outlier_columns (list): Columns for outlier removal
    normalize_method (str): Normalization method ('zscore', 'minmax', or None)
    missing_threshold (float): Threshold for removing high-missing columns
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Remove columns with high missing values
    high_missing = detect_missing_patterns(df_clean, missing_threshold)
    df_clean = df_clean.drop(columns=high_missing)
    
    # Remove outliers
    df_clean = remove_outliers_iqr(df_clean, outlier_columns)
    
    # Apply normalization
    if normalize_method == 'zscore':
        df_clean = zscore_normalize(df_clean)
    elif normalize_method == 'minmax':
        df_clean = minmax_normalize(df_clean)
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=10):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"Dataframe has less than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataframe is valid"