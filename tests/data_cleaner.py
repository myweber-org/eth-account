
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

def clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'constant').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if fill_missing and cleaned_df.isnull().sum().any():
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().sum() > 0:
                if strategy == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif strategy == 'median':
                    fill_value = cleaned_df[col].median()
                elif strategy == 'constant':
                    fill_value = 0
                else:
                    fill_value = cleaned_df[col].mean()
                
                cleaned_df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{col}' with {fill_value:.2f}")
        
        for col in categorical_cols:
            if cleaned_df[col].isnull().sum() > 0:
                mode_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown'
                cleaned_df[col].fillna(mode_value, inplace=True)
                print(f"Filled missing values in '{col}' with '{mode_value}'")
    
    print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
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
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 5],
        'B': [10, None, 10, 20, 30, 30],
        'C': ['x', 'y', 'x', None, 'z', 'z'],
        'D': [100, 200, 300, 400, 500, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning DataFrame...")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['A', 'B', 'C', 'D'], min_rows=3)
    print(f"\nDataFrame validation: {'Passed' if is_valid else 'Failed'}")
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        cleaned_data = self.data.copy()
        for col in columns:
            if col in cleaned_data.columns if hasattr(cleaned_data, 'columns') else True:
                col_data = cleaned_data[col] if hasattr(cleaned_data, 'columns') else cleaned_data[:, col]
                q1 = np.percentile(col_data, 25)
                q3 = np.percentile(col_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                if hasattr(cleaned_data, 'columns'):
                    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
                    cleaned_data = cleaned_data[mask]
                else:
                    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
                    cleaned_data = cleaned_data[mask, :]
        
        return cleaned_data
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        normalized_data = self.data.copy()
        for col in columns:
            if col in normalized_data.columns if hasattr(normalized_data, 'columns') else True:
                col_data = normalized_data[col] if hasattr(normalized_data, 'columns') else normalized_data[:, col]
                min_val = np.min(col_data)
                max_val = np.max(col_data)
                
                if max_val - min_val > 0:
                    normalized = (col_data - min_val) / (max_val - min_val)
                else:
                    normalized = col_data * 0
                
                if hasattr(normalized_data, 'columns'):
                    normalized_data[col] = normalized
                else:
                    normalized_data[:, col] = normalized
        
        return normalized_data
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        standardized_data = self.data.copy()
        for col in columns:
            if col in standardized_data.columns if hasattr(standardized_data, 'columns') else True:
                col_data = standardized_data[col] if hasattr(standardized_data, 'columns') else standardized_data[:, col]
                mean_val = np.mean(col_data)
                std_val = np.std(col_data)
                
                if std_val > 0:
                    standardized = (col_data - mean_val) / std_val
                else:
                    standardized = col_data * 0
                
                if hasattr(standardized_data, 'columns'):
                    standardized_data[col] = standardized
                else:
                    standardized_data[:, col] = standardized
        
        return standardized_data
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.data.columns if hasattr(self.data, 'columns') else range(self.data.shape[1])
        
        filled_data = self.data.copy()
        for col in columns:
            if col in filled_data.columns if hasattr(filled_data, 'columns') else True:
                col_data = filled_data[col] if hasattr(filled_data, 'columns') else filled_data[:, col]
                missing_mask = pd.isna(col_data) if hasattr(col_data, '__iter__') else pd.isna([col_data])[0]
                
                if np.any(missing_mask):
                    if strategy == 'mean':
                        fill_value = np.nanmean(col_data)
                    elif strategy == 'median':
                        fill_value = np.nanmedian(col_data)
                    elif strategy == 'mode':
                        fill_value = stats.mode(col_data[~missing_mask], keepdims=True)[0][0] if np.any(~missing_mask) else 0
                    else:
                        fill_value = 0
                    
                    if hasattr(filled_data, 'columns'):
                        filled_data.loc[missing_mask, col] = fill_value
                    else:
                        filled_data[missing_mask, col] = fill_value
        
        return filled_data