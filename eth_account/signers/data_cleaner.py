import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    missing_strategy (str): Strategy for missing values ('mean', 'median', 'drop').
    outlier_threshold (float): Number of standard deviations to define an outlier.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if missing_strategy == 'mean':
            cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
        elif missing_strategy == 'median':
            cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
        elif missing_strategy == 'drop':
            cleaned_df.dropna(subset=[col], inplace=True)
    
    # Remove outliers using Z-score method
    for col in numeric_cols:
        z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
        cleaned_df = cleaned_df[z_scores < outlier_threshold]
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'A': [1, 2, np.nan, 4, 100],
#         'B': [5, 6, 7, np.nan, 8],
#         'C': [9, 10, 11, 12, 13]
#     }
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataframe(df, missing_strategy='mean', outlier_threshold=2)
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
#     print(f"\nDataFrame validation: {is_valid}")import numpy as np
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
    
    def remove_outliers(self, columns):
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        return clean_df
    
    def impute_missing_mean(self, columns):
        filled_df = self.df.copy()
        for col in columns:
            if col in filled_df.columns and filled_df[col].isnull().any():
                mean_val = filled_df[col].mean()
                filled_df[col].fillna(mean_val, inplace=True)
        return filled_df
    
    def impute_missing_median(self, columns):
        filled_df = self.df.copy()
        for col in columns:
            if col in filled_df.columns and filled_df[col].isnull().any():
                median_val = filled_df[col].median()
                filled_df[col].fillna(median_val, inplace=True)
        return filled_df
    
    def drop_duplicates(self, subset=None):
        return self.df.drop_duplicates(subset=subset, keep='first')
    
    def get_cleaning_report(self):
        report = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.to_dict()
        }
        return report

def clean_dataset(df, outlier_columns=None, impute_strategy='mean'):
    cleaner = DataCleaner(df)
    
    if outlier_columns:
        df = cleaner.remove_outliers(outlier_columns)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if impute_strategy == 'mean':
        df = cleaner.impute_missing_mean(numeric_cols)
    elif impute_strategy == 'median':
        df = cleaner.impute_missing_median(numeric_cols)
    
    df = cleaner.drop_duplicates()
    
    return df, cleaner.get_cleaning_report()
import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    strategy (str): Strategy for missing value imputation ('mean', 'median', 'mode', 'drop').
    outlier_threshold (float): Number of standard deviations to consider as outlier.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()

    # Handle missing values
    if strategy == 'drop':
        df_clean = df_clean.dropna()
    elif strategy in ['mean', 'median', 'mode']:
        for column in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[column].isnull().any():
                if strategy == 'mean':
                    fill_value = df_clean[column].mean()
                elif strategy == 'median':
                    fill_value = df_clean[column].median()
                elif strategy == 'mode':
                    fill_value = df_clean[column].mode()[0]
                df_clean[column].fillna(fill_value, inplace=True)

    # Handle outliers using Z-score method
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
        df_clean = df_clean[z_scores < outlier_threshold]

    df_clean = df_clean.reset_index(drop=True)
    return df_clean

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
    if required_columns:
        if not all(col in df.columns for col in required_columns):
            return False

    if len(df) < min_rows:
        return False

    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    cleaned_df = clean_dataset(df, strategy='median', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned_df)

    is_valid = validate_data(cleaned_df, required_columns=['A', 'B', 'C'], min_rows=1)
    print(f"\nData validation passed: {is_valid}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if cleaned_df.isnull().sum().any():
        print("Handling missing values...")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values")
        else:
            for column in cleaned_df.select_dtypes(include=[np.number]).columns:
                if cleaned_df[column].isnull().any():
                    if fill_missing == 'mean':
                        fill_value = cleaned_df[column].mean()
                    elif fill_missing == 'median':
                        fill_value = cleaned_df[column].median()
                    elif fill_missing == 'mode':
                        fill_value = cleaned_df[column].mode()[0]
                    else:
                        raise ValueError("Invalid fill_missing method")
                    
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    print(f"Filled missing values in '{column}' with {fill_missing}: {fill_value:.2f}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for column in categorical_cols:
        if cleaned_df[column].isnull().any():
            mode_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown'
            cleaned_df[column] = cleaned_df[column].fillna(mode_value)
            print(f"Filled missing categorical values in '{column}' with mode: {mode_value}")
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if len(df) < min_rows:
        print(f"Validation failed: Dataset has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Validation failed: Missing required columns: {missing_columns}")
            return False
    
    if df.isnull().sum().sum() > 0:
        print("Warning: Dataset contains missing values")
    
    return True

def get_dataset_summary(df):
    """
    Generate a summary of the dataset.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        summary['categorical_stats'][col] = {
            'unique_values': df[col].nunique(),
            'top_value': df[col].mode()[0] if not df[col].mode().empty else None,
            'top_count': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        }
    
    return summary