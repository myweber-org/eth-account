
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].mean()
            )
        elif strategy == 'median':
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].median()
            )
        elif strategy == 'mode':
            for col in self.categorical_columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
        elif strategy == 'constant' and fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self

    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
            self.df = self.df[(z_scores < threshold).all(axis=1)]
        elif method == 'iqr':
            for col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self

    def normalize_data(self, method='minmax'):
        if method == 'minmax':
            for col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        elif method == 'standard':
            for col in self.numeric_columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        return self

    def get_cleaned_data(self):
        return self.df

def example_usage():
    data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['X', 'Y', np.nan, 'X', 'Y', 'Z']
    }
    df = pd.DataFrame(data)
    
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                  .handle_missing_values(strategy='mean')
                  .remove_outliers(method='zscore', threshold=2.5)
                  .normalize_data(method='minmax')
                  .get_cleaned_data())
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print("Cleaned Data:")
    print(result)
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_method (str or None): Method to fill missing values ('mean', 'median', 'mode', or None)
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method is not None:
        for column in cleaned_df.select_dtypes(include=['number']).columns:
            if cleaned_df[column].isnull().any():
                if fill_method == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_method == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_method == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    # Remove duplicates if specified
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values using specified strategy.
    """
    if columns is None:
        columns = df.columns
    
    df_filled = df.copy()
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            
            df_filled[col] = df[col].fillna(fill_value)
        else:
            df_filled[col] = df[col].fillna('Unknown')
    
    return df_filled

def normalize_column(df, column, method='minmax'):
    """
    Normalize specified column using given method.
    """
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df[column] = (df[column] - mean_val) / std_val
    
    return df

def clean_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to DataFrame.
    """
    if operations is None:
        operations = []
    
    cleaned_df = df.copy()
    
    for operation in operations:
        if operation['type'] == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, subset=operation.get('subset'))
        elif operation['type'] == 'fill_missing':
            cleaned_df = fill_missing_values(
                cleaned_df, 
                strategy=operation.get('strategy', 'mean'),
                columns=operation.get('columns')
            )
        elif operation['type'] == 'normalize':
            cleaned_df = normalize_column(
                cleaned_df,
                column=operation['column'],
                method=operation.get('method', 'minmax')
            )
    
    return cleaned_df

def validate_dataframe(df, rules=None):
    """
    Validate DataFrame against specified rules.
    """
    if rules is None:
        return True, []
    
    errors = []
    
    for rule in rules:
        if rule['type'] == 'not_null':
            null_count = df[rule['column']].isnull().sum()
            if null_count > 0:
                errors.append(f"Column {rule['column']} has {null_count} null values")
        
        elif rule['type'] == 'unique':
            duplicate_count = df[rule['column']].duplicated().sum()
            if duplicate_count > 0:
                errors.append(f"Column {rule['column']} has {duplicate_count} duplicate values")
        
        elif rule['type'] == 'range':
            min_val = rule.get('min')
            max_val = rule.get('max')
            
            if min_val is not None:
                below_min = (df[rule['column']] < min_val).sum()
                if below_min > 0:
                    errors.append(f"Column {rule['column']} has {below_min} values below minimum {min_val}")
            
            if max_val is not None:
                above_max = (df[rule['column']] > max_val).sum()
                if above_max > 0:
                    errors.append(f"Column {rule['column']} has {above_max} values above maximum {max_val}")
    
    return len(errors) == 0, errorsimport pandas as pd
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

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df = remove_outliers_iqr(df, col)
        
        print(f"After outlier removal: {df.shape}")
        
        for col in numeric_cols:
            df = normalize_minmax(df, col)
        
        cleaned_file = file_path.replace('.csv', '_cleaned.csv')
        df.to_csv(cleaned_file, index=False)
        print(f"Cleaned data saved to: {cleaned_file}")
        return df
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    sample_data.loc[np.random.choice(1000, 50), 'feature1'] = 500
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_dataset('sample_data.csv')
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
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
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range (IQR) method.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    column (int): Index of the column to process.
    
    Returns:
    numpy.ndarray: Data with outliers removed.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the cleaned data.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    if data.size == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan}
    
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data)
    }

def process_dataset(data_path, column_index):
    """
    Main function to load, clean, and analyze dataset.
    
    Parameters:
    data_path (str): Path to the data file (CSV format).
    column_index (int): Index of column to analyze.
    
    Returns:
    tuple: Cleaned data and statistics dictionary.
    """
    try:
        raw_data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
    cleaned_data = remove_outliers_iqr(raw_data, column_index)
    stats = calculate_statistics(cleaned_data[:, column_index])
    
    return cleaned_data, stats

if __name__ == "__main__":
    sample_data = np.array([
        [1.0, 10.2],
        [2.0, 15.3],
        [3.0, 100.5],
        [4.0, 12.8],
        [5.0, 11.1],
        [6.0, 200.7],
        [7.0, 13.4]
    ])
    
    cleaned, statistics = process_dataset("sample.csv", 1)
    
    if cleaned is not None:
        print(f"Original shape: {sample_data.shape}")
        print(f"Cleaned shape: {cleaned.shape}")
        print(f"Statistics: {statistics}")
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
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean.reset_index(drop=True)
        return removed_count
    
    def normalize_column(self, column, method='minmax'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if method == 'minmax':
            col_min = self.df[column].min()
            col_max = self.df[column].max()
            if col_max != col_min:
                self.df[f'{column}_normalized'] = (self.df[column] - col_min) / (col_max - col_min)
            else:
                self.df[f'{column}_normalized'] = 0.5
                
        elif method == 'zscore':
            mean = self.df[column].mean()
            std = self.df[column].std()
            if std > 0:
                self.df[f'{column}_normalized'] = (self.df[column] - mean) / std
            else:
                self.df[f'{column}_normalized'] = 0
                
        return self.df[f'{column}_normalized']
    
    def fill_missing(self, strategy='mean', custom_value=None):
        df_filled = self.df.copy()
        
        for col in df_filled.columns:
            if df_filled[col].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(df_filled[col]):
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_filled[col]):
                    df_filled[col] = df_filled[col].fillna(df_filled[col].median())
                elif strategy == 'mode':
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
                elif strategy == 'custom' and custom_value is not None:
                    df_filled[col] = df_filled[col].fillna(custom_value)
                elif strategy == 'ffill':
                    df_filled[col] = df_filled[col].fillna(method='ffill')
                elif strategy == 'bfill':
                    df_filled[col] = df_filled[col].fillna(method='bfill')
        
        self.df = df_filled
        return self.df.isnull().sum().sum()
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object', 'category']).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'A': np.random.normal(100, 15, 100),
        'B': np.random.exponential(50, 100),
        'C': np.random.randint(1, 100, 100),
        'category': np.random.choice(['X', 'Y', 'Z'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'A'] = np.nan
    df.loc[5, 'B'] = 1000
    
    cleaner = DataCleaner(df)
    print(f"Original shape: {cleaner.original_shape}")
    
    removed = cleaner.remove_outliers_iqr(['A', 'B'])
    print(f"Removed {removed} outliers")
    
    missing_filled = cleaner.fill_missing(strategy='mean')
    print(f"Filled {missing_filled} missing values")
    
    cleaner.normalize_column('A', method='zscore')
    cleaner.normalize_column('C', method='minmax')
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Summary: {summary}")
    
    return cleaned_df

if __name__ == "__main__":
    result_df = example_usage()
    print("Data cleaning completed successfully")import numpy as np
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
    if max_val - min_val == 0:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    numeric_cols = ['feature_a', 'feature_b']
    cleaned = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Outliers removed: {len(sample_data) - len(cleaned)}")