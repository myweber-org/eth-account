import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    remove_duplicates (bool): Whether to remove duplicate rows
    fill_method (str): Method to handle missing values - 'drop', 'mean', 'median', or 'mode'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_method == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_method == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
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

def get_data_summary(df):
    """
    Generate a summary of the DataFrame including missing values and data types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'unique_counts': df.nunique().to_dict()
    }
    
    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        summary['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    return summary

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'age': [25, 30, 35, None, 28, 28],
        'score': [85.5, 92.0, 78.5, 88.0, 95.5, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, remove_duplicates=True, fill_method='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, required_columns=['id', 'name', 'age', 'score'])
    print(f"Validation: {is_valid} - {message}")
    print("\n" + "="*50 + "\n")
    
    # Get summary
    summary = get_data_summary(cleaned)
    print("Data Summary:")
    print(f"Shape: {summary['shape']}")
    print(f"Missing values: {summary['missing_values']}")
import pandas as pd
import numpy as np
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
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        return self
        
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_norm = self.df.copy()
        for col in columns:
            if col in df_norm.columns:
                if method == 'zscore':
                    df_norm[col] = stats.zscore(df_norm[col])
                elif method == 'minmax':
                    df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
                elif method == 'robust':
                    median = df_norm[col].median()
                    iqr = df_norm[col].quantile(0.75) - df_norm[col].quantile(0.25)
                    df_norm[col] = (df_norm[col] - median) / iqr
        
        self.df = df_norm
        return self
        
    def fill_missing(self, columns=None, strategy='mean'):
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
                elif strategy == 'constant':
                    fill_value = 0
                else:
                    fill_value = df_filled[col].mean()
                
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'drop',
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and standardizing columns.
    
    Parameters:
    input_path: Path to input CSV file
    output_path: Path where cleaned CSV will be saved
    missing_strategy: Strategy for handling missing values ('drop', 'fill', 'interpolate')
    fill_value: Value to fill missing data with (if strategy is 'fill')
    
    Returns:
    Cleaned DataFrame
    """
    
    df = pd.read_csv(input_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'drop':
        df_cleaned = df.dropna(subset=numeric_columns)
    elif missing_strategy == 'fill':
        if fill_value is not None:
            df_cleaned = df.fillna({col: fill_value for col in numeric_columns})
        else:
            df_cleaned = df.fillna(df[numeric_columns].mean())
    elif missing_strategy == 'interpolate':
        df_cleaned = df.interpolate(method='linear', limit_direction='forward')
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    for col in numeric_columns:
        if df_cleaned[col].std() > 0:
            df_cleaned[col] = (df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std()
    
    df_cleaned.to_csv(output_path, index=False)
    
    print(f"Data cleaning completed. Original shape: {df.shape}, Cleaned shape: {df_cleaned.shape}")
    print(f"Cleaned data saved to: {output_path}")
    
    return df_cleaned

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame meets basic quality criteria.
    
    Parameters:
    df: DataFrame to validate
    
    Returns:
    Boolean indicating if DataFrame passes validation
    """
    
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if df.isnull().sum().sum() > len(df) * 0.5:
        print("Validation failed: Too many missing values")
        return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("Validation failed: No numeric columns found")
        return False
    
    for col in numeric_cols:
        if df[col].std() == 0:
            print(f"Warning: Column '{col}' has zero variance")
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': ['x', 'y', 'z', 'x', 'y']
    })
    
    sample_data.to_csv('sample_input.csv', index=False)
    
    cleaned_df = clean_csv_data(
        input_path='sample_input.csv',
        output_path='cleaned_output.csv',
        missing_strategy='fill',
        fill_value=0
    )
    
    is_valid = validate_dataframe(cleaned_df)
    print(f"Data validation result: {is_valid}")