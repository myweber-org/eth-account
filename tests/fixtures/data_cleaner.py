
import numpy as np
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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_values(df, strategy='mean'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'mode':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    return df

def clean_dataset(df, numeric_columns=None, outlier_removal=True, normalization='minmax'):
    df_clean = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = handle_missing_values(df_clean)
    
    if outlier_removal:
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean = remove_outliers_iqr(df_clean, col)
    
    for col in numeric_columns:
        if col in df_clean.columns:
            if normalization == 'minmax':
                df_clean = normalize_minmax(df_clean, col)
            elif normalization == 'zscore':
                df_clean = standardize_zscore(df_clean, col)
    
    return df_clean
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    factor (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize a column using min-max scaling to range [0, 1].
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result = data.copy()
    min_val = result[column].min()
    max_val = result[column].max()
    
    if max_val == min_val:
        result[column] = 0.5
    else:
        result[column] = (result[column] - min_val) / (max_val - min_val)
    
    return result

def standardize_zscore(data, column):
    """
    Standardize a column using z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to standardize
    
    Returns:
    pd.DataFrame: DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result = data.copy()
    mean_val = result[column].mean()
    std_val = result[column].std()
    
    if std_val == 0:
        result[column] = 0
    else:
        result[column] = (result[column] - mean_val) / std_val
    
    return result

def clean_dataset(data, numeric_columns, outlier_factor=1.5, normalize=True, standardize=False):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to process
    outlier_factor (float): IQR multiplier for outlier removal
    normalize (bool): Whether to apply min-max normalization
    standardize (bool): Whether to apply z-score standardization
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_factor)
    
    if normalize:
        for column in numeric_columns:
            if column in cleaned_data.columns:
                cleaned_data = normalize_minmax(cleaned_data, column)
    
    if standardize and not normalize:
        for column in numeric_columns:
            if column in cleaned_data.columns:
                cleaned_data = standardize_zscore(cleaned_data, column)
    
    return cleaned_data.reset_index(drop=True)

def validate_data(data, required_columns, allow_nan=False):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    data (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    allow_nan (bool): Whether NaN values are allowed
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    if not allow_nan:
        nan_columns = data.columns[data.isna().any()].tolist()
        if nan_columns:
            return False, f"NaN values found in columns: {nan_columns}"
    
    return True, "Data validation passed"import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
        
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            self.df[column] = (self.df[column] - self.df[column].min()) / (self.df[column].max() - self.df[column].min())
        elif method == 'zscore':
            self.df[column] = (self.df[column] - self.df[column].mean()) / self.df[column].std()
        return self
        
    def fill_missing(self, column, strategy='mean'):
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = strategy
            
        self.df[column].fillna(fill_value, inplace=True)
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
        
    def summary(self):
        print(f"Original rows: {self.original_shape[0]}")
        print(f"Cleaned rows: {self.df.shape[0]}")
        print(f"Rows removed: {self.get_removed_count()}")
        print(f"Columns: {self.df.shape[1]}")
        return self

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000)
    }
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature1'] = np.nan
    
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                 .remove_outliers_iqr('feature1')
                 .remove_outliers_zscore('feature2')
                 .fill_missing('feature1', 'mean')
                 .normalize_column('feature3', 'minmax')
                 .get_cleaned_data())
    
    cleaner.summary()
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print(result.head())import pandas as pd

def remove_duplicates(input_file, output_file, subset_columns=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        subset_columns (list, optional): Columns to consider for duplicate detection
    """
    try:
        df = pd.read_csv(input_file)
        
        if subset_columns:
            df_cleaned = df.drop_duplicates(subset=subset_columns, keep='first')
        else:
            df_cleaned = df.drop_duplicates(keep='first')
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        print(f"Removed {len(df) - len(df_cleaned)} duplicate rows")
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    remove_duplicates('raw_data.csv', 'cleaned_data.csv', ['id', 'email'])