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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.5)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_cleaning(df_before, df_after, column):
    stats_before = {
        'mean': df_before[column].mean(),
        'std': df_before[column].std(),
        'min': df_before[column].min(),
        'max': df_before[column].max()
    }
    stats_after = {
        'mean': df_after[column].mean(),
        'std': df_after[column].std(),
        'min': df_after[column].min(),
        'max': df_after[column].max()
    }
    return {'before': stats_before, 'after': stats_after}
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
        outlier_indices = set()
        for col in columns:
            if method == 'iqr':
                indices = self.detect_outliers_iqr(col, threshold)
                outlier_indices.update(indices)
        
        self.df = self.df.drop(index=list(outlier_indices)).reset_index(drop=True)
        removed_count = len(outlier_indices)
        return removed_count
    
    def normalize_column(self, column, method='zscore'):
        if method == 'zscore':
            self.df[f'{column}_normalized'] = stats.zscore(self.df[column])
        elif method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
        return self.df[f'{column}_normalized']
    
    def fill_missing(self, column, method='mean'):
        if method == 'mean':
            fill_value = self.df[column].mean()
        elif method == 'median':
            fill_value = self.df[column].median()
        elif method == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = method
        
        missing_count = self.df[column].isnull().sum()
        self.df[column] = self.df[column].fillna(fill_value)
        return missing_count
    
    def get_summary(self):
        current_shape = self.df.shape
        rows_removed = self.original_shape[0] - current_shape[0]
        return {
            'original_rows': self.original_shape[0],
            'current_rows': current_shape[0],
            'rows_removed': rows_removed,
            'columns': list(self.df.columns)
        }
    
    def get_clean_data(self):
        return self.df.copy()
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_data(df, columns, method='minmax'):
    normalized_df = df.copy()
    for col in columns:
        if col in df.columns:
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    normalized_df[col] = 0
            elif method == 'zscore':
                normalized_df[col] = stats.zscore(df[col])
    return normalized_df

def clean_dataset(df, numeric_columns):
    df_cleaned = df.dropna(subset=numeric_columns)
    df_no_outliers = remove_outliers_iqr(df_cleaned, numeric_columns)
    df_normalized = normalize_data(df_no_outliers, numeric_columns, method='minmax')
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 5, 100],
        'feature2': [10, 20, 30, 40, 50, 200],
        'category': ['A', 'B', 'A', 'B', 'A', 'B']
    }
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature1', 'feature2']
    result = clean_dataset(df, numeric_cols)
    print("Original dataset:")
    print(df)
    print("\nCleaned and normalized dataset:")
    print(result)