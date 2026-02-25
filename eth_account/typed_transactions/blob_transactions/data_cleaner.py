
import pandas as pd

def clean_dataframe(df, column, threshold, keep_above=True):
    """
    Filters a DataFrame based on a numeric threshold in a specified column.
    Returns a new DataFrame.
    """
    if keep_above:
        filtered_df = df[df[column] > threshold].copy()
    else:
        filtered_df = df[df[column] <= threshold].copy()

    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df

def remove_duplicates(df, subset=None):
    """
    Removes duplicate rows from the DataFrame.
    If subset is provided, only considers those columns for identifying duplicates.
    """
    cleaned_df = df.drop_duplicates(subset=subset, keep='first').copy()
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df

def main():
    # Example usage
    data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Alice'],
            'Score': [85, 92, 78, 45, 85],
            'Age': [25, 30, 35, 40, 25]}
    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)

    # Clean based on score threshold
    high_scorers = clean_dataframe(df, 'Score', 80, keep_above=True)
    print("\nDataFrame with scores > 80:")
    print(high_scorers)

    # Remove duplicates
    unique_df = remove_duplicates(df, subset=['Name', 'Score'])
    print("\nDataFrame after removing duplicates (based on Name and Score):")
    print(unique_df)

if __name__ == "__main__":
    main()
import pandas as pd

def clean_dataset(df, columns=None, drop_duplicates=True, drop_na=True):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns (list, optional): Specific columns to check for nulls/duplicates. 
                               If None, uses all columns.
    drop_duplicates (bool): Whether to drop duplicate rows.
    drop_na (bool): Whether to drop rows with null values.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if columns is None:
        columns = cleaned_df.columns
    
    if drop_na:
        cleaned_df = cleaned_df.dropna(subset=columns)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns, keep='first')
    
    return cleaned_df

def filter_by_value(df, column, value, keep=True):
    """
    Filter DataFrame rows based on column value.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to filter on.
    value: Value to filter by.
    keep (bool): If True, keep rows where column == value.
                 If False, keep rows where column != value.
    
    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    if keep:
        return df[df[column] == value].copy()
    else:
        return df[df[column] != value].copy()
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_column(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def clean_data(input_file, output_file):
    df = load_dataset(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    clean_data("raw_data.csv", "cleaned_data.csv")
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mean())
        elif strategy == 'median' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].median())
        elif strategy == 'mode' and self.categorical_columns.any():
            for col in self.categorical_columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
        elif strategy == 'constant' and fill_value is not None:
            self.df = self.df.fillna(fill_value)
        elif strategy == 'drop':
            self.df = self.df.dropna()
        return self

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        for col in columns:
            if col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self

    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        for col in columns:
            if col in self.numeric_columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std != 0:
                    self.df[col] = (self.df[col] - mean) / std
        return self

    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        for col in columns:
            if col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        return self

    def get_cleaned_data(self):
        return self.df

def example_usage():
    data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', np.nan, 'x', 'z']
    }
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                  .handle_missing_values(strategy='mean')
                  .remove_outliers_iqr(multiplier=1.5)
                  .standardize_data()
                  .get_cleaned_data())
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)

if __name__ == "__main__":
    example_usage()
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize column using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for col in columns:
        if col not in data_copy.columns:
            continue
            
        if data_copy[col].isnull().any():
            if strategy == 'mean':
                fill_value = data_copy[col].mean()
            elif strategy == 'median':
                fill_value = data_copy[col].median()
            elif strategy == 'mode':
                fill_value = data_copy[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_copy[col] = data_copy[col].fillna(fill_value)
    
    return data_copy

def validate_dataframe(data, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(data) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
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

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns using IQR or Z-score method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): Columns to process. If None, process all numeric columns.
    method (str): 'iqr' for interquartile range or 'zscore' for standard deviation.
    threshold (float): Threshold multiplier for IQR or Z-score cutoff.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    filtered_df = df.copy()
    
    for col in columns:
        if col not in df.columns or not np.issubdtype(df[col].dtype, np.number):
            continue
            
        col_data = df[col].dropna()
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        else:  # zscore
            mean = col_data.mean()
            std = col_data.std()
            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                mask = z_scores <= threshold
            else:
                mask = pd.Series(True, index=df.index)
        
        filtered_df = filtered_df[mask]
    
    return filtered_dfimport numpy as np
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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if df.empty:
        raise ValueError("DataFrame is empty")
    return True