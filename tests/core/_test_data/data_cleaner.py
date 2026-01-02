
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def z_score_normalize(data, column):
    """
    Normalize data using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def detect_missing_patterns(data, threshold=0.3):
    """
    Detect columns with missing values above threshold
    """
    missing_ratio = data.isnull().sum() / len(data)
    problematic_columns = missing_ratio[missing_ratio > threshold].index.tolist()
    
    return {
        'missing_ratios': missing_ratio.to_dict(),
        'problematic_columns': problematic_columns,
        'total_missing': data.isnull().sum().sum()
    }

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    removal_stats = {}
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data, removed = remove_outliers_iqr(cleaned_data, col, outlier_factor)
            removal_stats[col] = removed
            
            cleaned_data[col] = z_score_normalize(cleaned_data, col)
    
    missing_info = detect_missing_patterns(cleaned_data)
    
    return {
        'cleaned_data': cleaned_data,
        'outliers_removed': removal_stats,
        'missing_info': missing_info,
        'original_shape': data.shape,
        'cleaned_shape': cleaned_data.shape
    }

def validate_data_types(data, expected_types):
    """
    Validate column data types against expected types
    """
    validation_results = {}
    
    for col, expected_type in expected_types.items():
        if col in data.columns:
            actual_type = str(data[col].dtype)
            is_valid = actual_type == expected_type
            validation_results[col] = {
                'expected': expected_type,
                'actual': actual_type,
                'valid': is_valid
            }
    
    all_valid = all(result['valid'] for result in validation_results.values())
    
    return {
        'validation_results': validation_results,
        'all_valid': all_valid
    }
import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.cleaned_data = None
        
    def remove_outliers_iqr(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
    
    def normalize_minmax(self, column):
        min_val = self.data[column].min()
        max_val = self.data[column].max()
        self.data[column] = (self.data[column] - min_val) / (max_val - min_val)
        return self.data
    
    def fill_missing_mean(self, column):
        mean_val = self.data[column].mean()
        self.data[column].fillna(mean_val, inplace=True)
        return self.data
    
    def clean_pipeline(self, columns_to_clean):
        for col in columns_to_clean:
            self.data = self.remove_outliers_iqr(col)
            self.data = self.normalize_minmax(col)
            self.data = self.fill_missing_mean(col)
        self.cleaned_data = self.data
        return self.cleaned_data
    
    def save_cleaned_data(self, filename):
        if self.cleaned_data is not None:
            self.cleaned_data.to_csv(filename, index=False)
            return True
        return False