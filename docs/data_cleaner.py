
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset)
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self.df
    
    def handle_missing_values(self, strategy: str = 'drop', fill_value: Optional[float] = None) -> pd.DataFrame:
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                self.df = self.df.fillna(self.df.mean())
        print(f"Missing values handled using '{strategy}' strategy")
        return self.df
    
    def normalize_column(self, column: str) -> pd.DataFrame:
        if column in self.df.columns:
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            if col_max != col_min:
                self.df[column] = (self.df[column] - col_min) / (col_max - col_min)
        return self.df
    
    def get_summary(self) -> dict:
        return {
            'original_rows': self.original_shape[0],
            'current_rows': len(self.df),
            'original_columns': self.original_shape[1],
            'current_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates_removed': self.original_shape[0] - len(self.df)
        }

def clean_dataset(file_path: str, output_path: str) -> None:
    df = pd.read_csv(file_path)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates()
    cleaner.handle_missing_values(strategy='fill')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaner.normalize_column(col)
    
    cleaner.df.to_csv(output_path, index=False)
    summary = cleaner.get_summary()
    
    print(f"Data cleaning complete. Results saved to {output_path}")
    print(f"Summary: {summary}")
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset)
        return self
        
    def handle_missing_values(self, 
                             strategy: str = 'mean',
                             columns: Optional[List[str]] = None) -> 'DataCleaner':
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
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
            self.df[col] = self.df[col].fillna(fill_value)
            
        return self
        
    def remove_outliers_iqr(self, 
                           columns: Optional[List[str]] = None,
                           multiplier: float = 1.5) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in columns:
            if col not in self.df.columns:
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            
        return self
        
    def normalize_columns(self, 
                         columns: Optional[List[str]] = None,
                         method: str = 'minmax') -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == 'minmax':
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
            elif method == 'zscore':
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val != 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
                    
        return self
        
    def encode_categorical(self, 
                          columns: Optional[List[str]] = None,
                          method: str = 'onehot') -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == 'onehot':
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                
            elif method == 'label':
                self.df[col] = self.df[col].astype('category').cat.codes
                
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df.copy()
        
    def get_summary(self) -> Dict[str, Union[int, float]]:
        cleaned_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': self.original_shape[0] - cleaned_shape[0],
            'columns_changed': abs(self.original_shape[1] - cleaned_shape[1])
        }

def load_and_clean_csv(filepath: str, 
                      cleaning_steps: Optional[List[Dict]] = None) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        for step in cleaning_steps:
            method_name = step.get('method')
            params = step.get('params', {})
            
            if hasattr(cleaner, method_name):
                method = getattr(cleaner, method_name)
                method(**params)
    
    return cleaner.get_cleaned_data()
import pandas as pd
import numpy as np
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

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df = df.dropna()
    return df

def calculate_statistics(df, column):
    return {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'skewness': stats.skew(df[column].dropna()),
        'kurtosis': stats.kurtosis(df[column].dropna())
    }

if __name__ == "__main__":
    cleaned_data = clean_dataset('sample_data.csv', ['age', 'income', 'score'])
    stats_result = calculate_statistics(cleaned_data, 'income')
    print(f"Dataset cleaned. Statistics: {stats_result}")
    cleaned_data.to_csv('cleaned_data.csv', index=False)
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_zscore'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(filepath, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    df = load_data(filepath)
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            df = remove_outliers_iqr(df, col)
        elif outlier_method == 'zscore':
            df = remove_outliers_zscore(df, col)
        
        if normalize_method == 'minmax':
            df = normalize_minmax(df, col)
        elif normalize_method == 'zscore':
            df = normalize_zscore(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset(
        'raw_data.csv',
        ['age', 'income', 'score'],
        outlier_method='zscore',
        normalize_method='zscore'
    )
    cleaned_df.to_csv('cleaned_data.csv', index=False)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
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
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
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
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nValidation: {is_valid} - {message}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
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

def z_score_normalize(data, column):
    """
    Normalize data using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean = data[column].mean()
    std = data[column].std()
    
    if std == 0:
        return data[column]
    
    normalized = (data[column] - mean) / std
    return normalized

def min_max_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column]
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    
    if feature_range != (0, 1):
        min_target, max_target = feature_range
        normalized = normalized * (max_target - min_target) + min_target
    
    return normalized

def detect_skewed_columns(data, threshold=0.5):
    """
    Detect columns with skewed distributions
    """
    skewed_columns = []
    
    for column in data.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(data[column].dropna())
        if abs(skewness) > threshold:
            skewed_columns.append((column, skewness))
    
    return sorted(skewed_columns, key=lambda x: abs(x[1]), reverse=True)

def apply_log_transform(data, column):
    """
    Apply log transformation to reduce skewness
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if data[column].min() <= 0:
        shifted_data = data[column] - data[column].min() + 1
        return np.log(shifted_data)
    
    return np.log(data[column])

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5, normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    cleaning_report = {}
    
    for column in numeric_columns:
        if column not in data.columns:
            continue
            
        original_count = len(cleaned_data)
        
        cleaned_data, removed = remove_outliers_iqr(cleaned_data, column, outlier_factor)
        cleaning_report[column] = {
            'outliers_removed': removed,
            'percentage_removed': (removed / original_count) * 100
        }
        
        if normalize_method == 'zscore':
            cleaned_data[f'{column}_normalized'] = z_score_normalize(cleaned_data, column)
        elif normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = min_max_normalize(cleaned_data, column)
    
    skewed_cols = detect_skewed_columns(cleaned_data[numeric_columns])
    cleaning_report['skewed_columns'] = skewed_cols
    
    for column, skewness in skewed_cols:
        if abs(skewness) > 1:
            cleaned_data[f'{column}_log'] = apply_log_transform(cleaned_data, column)
    
    return cleaned_data, cleaning_report

def save_cleaning_report(report, filename='cleaning_report.json'):
    """
    Save cleaning report to JSON file
    """
    import json
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    
    return filename
import pandas as pd
import numpy as np
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

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df = df.dropna()
    return df

def calculate_statistics(df, column):
    stats_dict = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'skewness': stats.skew(df[column].dropna()),
        'kurtosis': stats.kurtosis(df[column].dropna())
    }
    return stats_dict

if __name__ == "__main__":
    data_path = 'sample_data.csv'
    numeric_cols = ['age', 'income', 'score']
    
    cleaned_data = clean_dataset(data_path, numeric_cols)
    print(f"Original shape: {pd.read_csv(data_path).shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    
    for col in numeric_cols:
        if col in cleaned_data.columns:
            stats_result = calculate_statistics(cleaned_data, col)
            print(f"\nStatistics for {col}:")
            for key, value in stats_result.items():
                print(f"{key}: {value:.4f}")
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def detect_outliers_iqr(self, column, threshold=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers.index.tolist()
    
    def remove_outliers(self, columns, method='iqr', threshold=1.5):
        outlier_indices = []
        for col in columns:
            if method == 'iqr':
                indices = self.detect_outliers_iqr(col, threshold)
                outlier_indices.extend(indices)
        
        unique_indices = list(set(outlier_indices))
        self.df = self.df.drop(index=unique_indices)
        return len(unique_indices)
    
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val - min_val != 0:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val != 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
        return self.df[column]
    
    def fill_missing(self, column, method='mean'):
        if method == 'mean':
            fill_value = self.df[column].mean()
        elif method == 'median':
            fill_value = self.df[column].median()
        elif method == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = method
        
        self.df[column] = self.df[column].fillna(fill_value)
        return self.df[column]
    
    def get_cleaned_data(self):
        return self.df
    
    def get_cleaning_report(self):
        cleaned_shape = self.df.shape
        rows_removed = self.original_shape[0] - cleaned_shape[0]
        cols_removed = self.original_shape[1] - cleaned_shape[1]
        
        report = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'missing_values': self.df.isnull().sum().sum()
        }
        return report

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.iloc[5:10, 0] = np.nan
    df.iloc[15:20, 1] = np.nan
    
    df.loc[95, 'feature_a'] = 500
    df.loc[96, 'feature_b'] = 300
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", cleaner.original_shape)
    print("Missing values:", sample_df.isnull().sum().sum())
    
    cleaner.fill_missing('feature_a', 'mean')
    cleaner.fill_missing('feature_b', 'median')
    
    outliers_removed = cleaner.remove_outliers(['feature_a', 'feature_b'])
    print(f"Removed {outliers_removed} outliers")
    
    cleaner.normalize_column('feature_a', 'minmax')
    cleaner.normalize_column('feature_b', 'zscore')
    
    cleaned_df = cleaner.get_cleaned_data()
    report = cleaner.get_cleaning_report()
    
    print("\nCleaning Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    print("\nCleaned data statistics:")
    print(cleaned_df.describe())
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> 'DataCleaner':
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
                fill_value = self.df[col].mode()[0]
            elif strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
            else:
                fill_value = 0
                
            if strategy != 'drop':
                self.df[col] = self.df[col].fillna(fill_value)
                
        return self
    
    def convert_types(self, type_map: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_map.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert {col} to {dtype}: {e}")
        return self
    
    def remove_outliers(self, column: str, method: str = 'iqr', threshold: float = 1.5) -> 'DataCleaner':
        if column not in self.df.columns:
            return self
            
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.df[column]))
            self.df = self.df[z_scores < threshold]
            
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_summary(self) -> Dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }

def clean_csv_file(input_path: str, output_path: str, **kwargs) -> Dict:
    try:
        df = pd.read_csv(input_path)
        cleaner = DataCleaner(df)
        
        if 'missing_strategy' in kwargs:
            cleaner.handle_missing_values(strategy=kwargs['missing_strategy'])
        
        if 'type_map' in kwargs:
            cleaner.convert_types(kwargs['type_map'])
            
        cleaned_df = cleaner.get_cleaned_data()
        cleaned_df.to_csv(output_path, index=False)
        
        return {
            'success': True,
            'summary': cleaner.get_summary(),
            'output_file': output_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }import pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df: pd.DataFrame, 
                         column_type_map: dict) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Input DataFrame
        column_type_map: Dictionary mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_copy = df.copy()
    for column, dtype in column_type_map.items():
        if column in df_copy.columns:
            try:
                df_copy[column] = df_copy[column].astype(dtype)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert column '{column}' to {dtype}")
    return df_copy

def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'drop',
                         fill_value: Union[int, float, str] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: Value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_filled = df.copy()
            df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())
            return df_filled
    else:
        raise ValueError("Strategy must be either 'drop' or 'fill'")

def clean_dataframe(df: pd.DataFrame,
                   deduplicate: bool = True,
                   type_conversions: dict = None,
                   missing_strategy: str = 'drop') -> pd.DataFrame:
    """
    Main function to clean DataFrame with multiple operations.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        type_conversions: Dictionary for column type conversions
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None],
        'age': [25, 30, 30, None, 35],
        'score': ['85', '90', '90', '95', '100']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    type_map = {'score': 'int32'}
    cleaned = clean_dataframe(df, 
                             type_conversions=type_map,
                             missing_strategy='fill')
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nCleaned DataFrame info:")
    print(cleaned.info())
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> 'DataCleaner':
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
                fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else np.nan
            elif strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            if strategy != 'drop':
                self.df[col] = self.df[col].fillna(fill_value)
        
        return self
    
    def convert_types(self, type_map: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_map.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    elif dtype == 'category':
                        self.df[col] = self.df[col].astype('category')
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert {col} to {dtype}: {e}")
        
        return self
    
    def remove_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_cleaning_report(self) -> Dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values_remaining': self.df.isnull().sum().sum()
        }

def clean_csv_file(filepath: str, output_path: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.convert_types({
        'date': 'datetime',
        'category': 'category'
    })
    cleaner.remove_outliers()
    
    cleaned_df = cleaner.get_cleaned_data()
    report = cleaner.get_cleaning_report()
    
    print(f"Data cleaning completed:")
    print(f"  - Removed {report['rows_removed']} rows")
    print(f"  - Remaining missing values: {report['missing_values_remaining']}")
    
    if output_path:
        cleaned_df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return cleaned_df
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: Input pandas DataFrame
        column_mapping: Dictionary to rename columns (optional)
        drop_duplicates: Whether to remove duplicate rows
        normalize_text: Whether to normalize text columns
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    return cleaned_df

def _normalize_string(text):
    """Normalize a string by converting to lowercase and removing extra whitespace."""
    if pd.isna(text):
        return text
    
    normalized = str(text).lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Args:
        df: Input DataFrame
        email_column: Name of the column containing email addresses
    
    Returns:
        DataFrame with validation results
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validation_results = df.copy()
    validation_results['email_valid'] = validation_results[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    return validation_results