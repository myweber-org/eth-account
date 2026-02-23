
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self.df
    
    def normalize_column(self, column_name: str, method: str = 'minmax') -> pd.DataFrame:
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
        
        if method == 'minmax':
            col_min = self.df[column_name].min()
            col_max = self.df[column_name].max()
            if col_max != col_min:
                self.df[f"{column_name}_normalized"] = (self.df[column_name] - col_min) / (col_max - col_min)
            else:
                self.df[f"{column_name}_normalized"] = 0
        elif method == 'zscore':
            col_mean = self.df[column_name].mean()
            col_std = self.df[column_name].std()
            if col_std > 0:
                self.df[f"{column_name}_normalized"] = (self.df[column_name] - col_mean) / col_std
            else:
                self.df[f"{column_name}_normalized"] = 0
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
        
        return self.df
    
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if strategy == 'mean':
                fill_value = self.df[col].mean()
            elif strategy == 'median':
                fill_value = self.df[col].median()
            elif strategy == 'mode':
                fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
            elif strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
                continue
            else:
                raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
            
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                self.df[col] = self.df[col].fillna(fill_value)
                print(f"Filled {missing_count} missing values in column '{col}' with {strategy}: {fill_value}")
        
        return self.df
    
    def get_summary(self) -> dict:
        return {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'columns': list(self.df.columns),
            'missing_values': self.df.isna().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict()
        }
    
    def save_cleaned_data(self, filepath: str):
        self.df.to_csv(filepath, index=False)
        print(f"Cleaned data saved to {filepath}")
import pandas as pd
import re

def clean_dataset(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_names (list): List of column names to normalize.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Normalize specified string columns
    for col in column_names:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].apply(
                lambda x: re.sub(r'\s+', ' ', str(x).strip().lower()) if pd.notnull(x) else x
            )
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with valid emails and a validation flag.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    df_validated = df.copy()
    df_validated['email_valid'] = df_validated[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    return df_validated

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson  '],
        'email': ['john@example.com', 'jane@example', 'john@example.com', 'bob@example.org'],
        'age': [25, 30, 25, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the dataset
    cleaned = clean_dataset(df, ['name', 'email'])
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    # Validate emails
    validated = validate_email_column(cleaned, 'email')
    print("Validated DataFrame:")
    print(validated)
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'constant':
            if fill_value is not None:
                self.df.fillna(fill_value, inplace=True)
            else:
                raise ValueError("fill_value must be provided for constant strategy")
        else:
            raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'constant'")
        
        for col in self.categorical_columns:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        return self.df

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        df_clean = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        return self.df

    def remove_duplicates(self, subset=None, keep='first'):
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self.df

    def normalize_data(self, columns=None, method='minmax'):
        if columns is None:
            columns = self.numeric_columns
        
        df_normalized = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                if method == 'minmax':
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    if max_val != min_val:
                        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean_val = df_normalized[col].mean()
                    std_val = df_normalized[col].std()
                    if std_val != 0:
                        df_normalized[col] = (df_normalized[col] - mean_val) / std_val
                else:
                    raise ValueError("Invalid method. Choose from 'minmax' or 'zscore'")
        
        self.df = df_normalized
        return self.df

    def get_cleaned_data(self):
        return self.df.copy()

def load_and_clean_data(file_path, cleaning_steps=None):
    df = pd.read_csv(file_path)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        for step in cleaning_steps:
            if step['function'] == 'handle_missing_values':
                cleaner.handle_missing_values(**step.get('params', {}))
            elif step['function'] == 'remove_outliers':
                cleaner.remove_outliers_iqr(**step.get('params', {}))
            elif step['function'] == 'remove_duplicates':
                cleaner.remove_duplicates(**step.get('params', {}))
            elif step['function'] == 'normalize_data':
                cleaner.normalize_data(**step.get('params', {}))
    
    return cleaner.get_cleaned_data()
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (np.ndarray): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    np.ndarray: Data with outliers removed from the specified column.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1] or column < 0:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]

def example_usage():
    # Generate sample data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(50, 10, 90)
    outlier_data = np.random.uniform(100, 200, 10)
    sample_data = np.concatenate([normal_data, outlier_data])
    sample_data = sample_data.reshape(-1, 1)
    
    print(f"Original data shape: {sample_data.shape}")
    cleaned_data = remove_outliers_iqr(sample_data, 0)
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Number of outliers removed: {sample_data.shape[0] - cleaned_data.shape[0]}")

if __name__ == "__main__":
    example_usage()import pandas as pd

def clean_dataset(df, columns_to_check=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df: pandas DataFrame to clean
        columns_to_check: list of columns to check for duplicates (None for all columns)
        drop_duplicates: boolean indicating whether to drop duplicate rows
        fill_missing: strategy for filling missing values ('mean', 'median', 'mode', or value)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    
    for column in cleaned_df.columns:
        if cleaned_df[column].dtype in ['int64', 'float64']:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
                else:
                    cleaned_df[column].fillna(fill_missing, inplace=True)
        else:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
                else:
                    cleaned_df[column].fillna(fill_missing, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of columns that must be present
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Remove outliers
    if outlier_method == 'iqr':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
    elif outlier_method == 'zscore':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    # Normalize data
    if normalize_method == 'minmax':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
    elif normalize_method == 'zscore':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        allow_nan: whether to allow NaN values
    
    Returns:
        tuple: (is_valid, message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if required_columns:
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            return False, f"Missing required columns: {missing}"
    
    if not allow_nan and data.isnull().any().any():
        nan_cols = data.columns[data.isnull().any()].tolist()
        return False, f"NaN values found in columns: {nan_cols}"
    
    return True, "Data validation passed"
import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Clean a text column by removing extra whitespace, converting to lowercase,
    and removing special characters.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
    
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from the DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_numeric_column(df, column_name):
    """
    Normalize a numeric column to have values between 0 and 1.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise ValueError(f"Column '{column_name}' is not numeric")
    
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    
    if max_val == min_val:
        df[column_name] = 0.5
    else:
        df[column_name] = (df[column_name] - min_val) / (max_val - min_val)
    
    return df

def process_dataframe(df, text_columns=None, numeric_columns=None, deduplicate=True):
    """
    Main function to process a DataFrame with multiple cleaning steps.
    """
    if text_columns:
        for col in text_columns:
            df = clean_text_column(df, col)
    
    if numeric_columns:
        for col in numeric_columns:
            df = normalize_numeric_column(df, col)
    
    if deduplicate:
        df = remove_duplicates(df)
    
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Alice Johnson  '],
        'age': [25, 30, 25, 35],
        'email': ['JOHN@example.com', 'jane@test.com', 'john@example.com', 'alice@demo.net'],
        'score': [85, 92, 85, 78]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    processed_df = process_dataframe(
        df,
        text_columns=['name', 'email'],
        numeric_columns=['age', 'score'],
        deduplicate=True
    )
    
    print("Processed DataFrame:")
    print(processed_df)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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
    If columns is None, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"
import pandas as pd
import numpy as np
from typing import Optional, List

def clean_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    handle_nulls: str = 'drop',
    fill_value: Optional[float] = None,
    columns_to_clean: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by removing duplicates and handling null values.
    
    Parameters:
    df: Input DataFrame
    drop_duplicates: Whether to drop duplicate rows
    handle_nulls: Strategy for handling nulls - 'drop', 'fill', or 'ignore'
    fill_value: Value to use when filling nulls (required if handle_nulls='fill')
    columns_to_clean: Specific columns to apply cleaning to (None for all columns)
    
    Returns:
    Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns_to_clean:
        df_subset = df_clean[columns_to_clean]
    else:
        df_subset = df_clean
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if handle_nulls == 'drop':
        df_clean = df_clean.dropna(subset=columns_to_clean)
    elif handle_nulls == 'fill':
        if fill_value is None:
            raise ValueError("fill_value must be provided when handle_nulls='fill'")
        df_clean[columns_to_clean] = df_subset.fillna(fill_value)
    
    return df_clean

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns and has no negative values in numeric columns.
    
    Parameters:
    df: DataFrame to validate
    required_columns: List of column names that must be present
    
    Returns:
    Boolean indicating if validation passed
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    negative_check = (df[numeric_cols] < 0).any().any()
    
    if negative_check:
        print("Warning: DataFrame contains negative values in numeric columns")
    
    return True

def calculate_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate basic statistics for numeric columns in DataFrame.
    
    Parameters:
    df: Input DataFrame
    
    Returns:
    Dictionary containing statistics for each numeric column
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
            'null_count': df[col].isnull().sum()
        }
    
    return stats

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, 20.3, np.nan, 15.7, 10.5, 10.5],
        'category': ['A', 'B', 'A', 'C', 'A', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(
        df,
        drop_duplicates=True,
        handle_nulls='fill',
        fill_value=0.0,
        columns_to_clean=['value']
    )
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    if validate_dataframe(cleaned_df, ['id', 'value', 'category']):
        stats = calculate_statistics(cleaned_df)
        print("Statistics:")
        for col, col_stats in stats.items():
            print(f"{col}: {col_stats}")
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for identifying duplicates
        keep: determines which duplicates to keep ('first', 'last', or False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
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

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"