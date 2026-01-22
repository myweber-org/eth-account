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