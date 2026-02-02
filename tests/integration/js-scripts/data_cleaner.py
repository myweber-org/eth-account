
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: A list containing elements that may have duplicates.
    
    Returns:
        A new list with duplicates removed.
    """
    seen = set()
    result = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_strings(data_list):
    """
    Convert string representations of numbers to integers where possible.
    
    Args:
        data_list: A list containing mixed types.
    
    Returns:
        A new list with numeric strings converted to integers.
    """
    result = []
    for item in data_list:
        if isinstance(item, str) and item.isdigit():
            result.append(int(item))
        else:
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, "4", "4", 5, "hello", 3]
    cleaned = remove_duplicates(sample_data)
    final_data = clean_numeric_strings(cleaned)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {final_data}")
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'constant':
            if fill_value is not None:
                self.df.fillna(fill_value, inplace=True)
            else:
                raise ValueError("fill_value must be provided for constant strategy")
        elif strategy == 'drop':
            self.df.dropna(inplace=True)
        else:
            raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', 'constant', or 'drop'")
        
        for col in self.categorical_columns:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        return self.df

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        cleaned_df = self.df.copy()
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
        
        self.df = cleaned_df
        return self.df

    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
        
        return self.df

    def normalize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        return self.df

    def get_cleaned_data(self):
        return self.df.copy()

def load_and_clean_data(filepath, missing_strategy='mean', remove_outliers=True):
    try:
        df = pd.read_csv(filepath)
        cleaner = DataCleaner(df)
        cleaner.handle_missing_values(strategy=missing_strategy)
        
        if remove_outliers:
            cleaner.remove_outliers_iqr()
        
        return cleaner.get_cleaned_data()
    except Exception as e:
        print(f"Error loading or cleaning data: {e}")
        return None
import pandas as pd
import numpy as np

def remove_outliers(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return df[z_scores < threshold]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')