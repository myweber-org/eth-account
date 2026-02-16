
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Remove outliers using z-score method
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mean())
        elif strategy == 'median' and self.numeric_columns:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].median())
        elif strategy == 'mode' and self.categorical_columns:
            for col in self.categorical_columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
        elif strategy == 'constant' and fill_value is not None:
            self.df.fillna(fill_value, inplace=True)
        elif strategy == 'drop':
            self.df.dropna(inplace=True)
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
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
        return self

    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        for col in columns:
            if col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        return self

    def get_cleaned_data(self):
        return self.df

def example_usage():
    data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', np.nan, 'x', 'y']
    }
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                  .handle_missing_values(strategy='mean')
                  .remove_outliers_iqr(multiplier=1.5)
                  .standardize_data()
                  .get_cleaned_data())
    print(cleaned_df)

if __name__ == "__main__":
    example_usage()
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                print(f"Filled missing values in column '{col}' with median")
        
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col] = cleaned_df[col].fillna('Unknown')
                print(f"Filled missing values in column '{col}' with 'Unknown'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    print(f"DataFrame validated: {df.shape[0]} rows, {df.shape[1]} columns")
    return True

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
        'age': [25, 30, 30, None, 35, 28],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for detecting outliers ('iqr', 'zscore')
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if missing_strategy != 'drop':
            if missing_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif missing_strategy == 'median':
                fill_value = df_clean[col].median()
            elif missing_strategy == 'mode':
                fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else np.nan
            else:
                fill_value = 0
            df_clean[col].fillna(fill_value, inplace=True)
        else:
            df_clean = df_clean.dropna(subset=[col])
        
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = np.where((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), 
                                     df_clean[col].median(), df_clean[col])
        elif outlier_method == 'zscore':
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            z_scores = (df_clean[col] - mean_val) / std_val
            df_clean[col] = np.where(np.abs(z_scores) > 3, mean_val, df_clean[col])
    
    return df_clean

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list): Columns to consider for identifying duplicates
    keep (str): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to normalize
    method (str): Normalization method ('minmax', 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    df_norm = df.copy()
    
    if columns is None:
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df_norm.columns:
            continue
            
        if method == 'minmax':
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val != min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val != 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, 8, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, missing_strategy='mean', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    normalized_df = normalize_data(cleaned_df, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized_df)