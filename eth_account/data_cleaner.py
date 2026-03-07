import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'constant' and fill_value is not None:
            self.df.fillna(fill_value, inplace=True)
        return self
    
    def remove_outliers_zscore(self, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
        outlier_mask = (z_scores < threshold).all(axis=1)
        self.df = self.df[outlier_mask]
        return self
    
    def remove_outliers_iqr(self, multiplier=1.5):
        for col in self.numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self
    
    def normalize_data(self, method='minmax'):
        if method == 'minmax':
            for col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        elif method == 'standard':
            for col in self.numeric_columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()

def example_usage():
    data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, np.nan, 7, 8, 9],
        'C': ['x', 'y', 'x', 'y', 'x']
    }
    df = pd.DataFrame(data)
    
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                 .handle_missing_values(strategy='mean')
                 .remove_outliers_zscore(threshold=2.5)
                 .normalize_data(method='minmax')
                 .get_cleaned_data())
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print("Original shape:", pd.DataFrame({
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, np.nan, 7, 8, 9],
        'C': ['x', 'y', 'x', 'y', 'x']
    }).shape)
    print("Cleaned shape:", result.shape)
    print("Cleaned data:\n", result)
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_method (str or None): Method to fill missing values. 
                                   Options: 'mean', 'median', 'mode', or None for dropping.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method is None:
        cleaned_df = cleaned_df.dropna()
    else:
        if fill_method == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_method == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_method == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        else:
            raise ValueError("fill_method must be 'mean', 'median', 'mode', or None")
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, message)
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

# Example usage
if __name__ == "__main__":
    # Create sample data with some issues
    sample_data = {
        'A': [1, 2, None, 4, 2],
        'B': [5, None, 7, 8, 5],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    # Clean the data
    cleaned = clean_dataset(df, drop_duplicates=True, fill_method='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nShape:", cleaned.shape)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B', 'C'], min_rows=1)
    print(f"\nValidation: {is_valid} - {message}")