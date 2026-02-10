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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def export_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data exported to {output_path}")
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    def detect_outliers_iqr(self, column, threshold=1.5):
        if column not in self.numeric_columns:
            raise ValueError(f"Column {column} is not numeric")
            
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers.index.tolist()
    
    def remove_outliers(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.numeric_columns
            
        outlier_indices = set()
        for col in columns:
            if col in self.numeric_columns:
                indices = self.detect_outliers_iqr(col, threshold)
                outlier_indices.update(indices)
        
        clean_df = self.df.drop(index=list(outlier_indices))
        return clean_df
    
    def impute_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        imputed_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns and imputed_df[col].isnull().any():
                median_val = imputed_df[col].median()
                imputed_df[col].fillna(median_val, inplace=True)
        
        return imputed_df
    
    def standardize_numeric(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        standardized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean_val = standardized_df[col].mean()
                std_val = standardized_df[col].std()
                if std_val > 0:
                    standardized_df[col] = (standardized_df[col] - mean_val) / std_val
        
        return standardized_df
    
    def get_summary(self):
        summary = {
            'original_rows': len(self.df),
            'original_columns': len(self.df.columns),
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df.isnull().sum().sum(),
            'missing_percentage': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        }
        return summary

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    print("Data Summary:")
    summary = cleaner.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    if summary['missing_values'] > 0:
        df = cleaner.impute_missing_median()
        print(f"Imputed {summary['missing_values']} missing values")
    
    df = cleaner.remove_outliers(threshold=1.5)
    print(f"Removed outliers using IQR method")
    
    df = cleaner.standardize_numeric()
    print("Standardized numeric columns")
    
    return df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [1.1, 2.2, np.nan, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
        'category': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y']
    })
    
    cleaner = DataCleaner(sample_data)
    print("Sample data cleaning demonstration:")
    print("Original data shape:", sample_data.shape)
    
    clean_data = cleaner.remove_outliers()
    print("After outlier removal:", clean_data.shape)
    
    imputed_data = cleaner.impute_missing_median()
    print("After missing value imputation:", imputed_data.shape)
    
    standardized_data = cleaner.standardize_numeric()
    print("After standardization - mean of column A:", standardized_data['A'].mean())