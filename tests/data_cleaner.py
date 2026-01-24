
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns
    
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
        elif strategy == 'custom' and fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self
    
    def remove_outliers_zscore(self, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
        outlier_mask = (z_scores < threshold).all(axis=1)
        self.df = self.df[outlier_mask]
        return self
    
    def remove_outliers_iqr(self, multiplier=1.5):
        for col in self.numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self
    
    def standardize_data(self):
        self.df[self.numeric_columns] = (
            self.df[self.numeric_columns] - self.df[self.numeric_columns].mean()
        ) / self.df[self.numeric_columns].std()
        return self
    
    def normalize_data(self):
        self.df[self.numeric_columns] = (
            self.df[self.numeric_columns] - self.df[self.numeric_columns].min()
        ) / (self.df[self.numeric_columns].max() - self.df[self.numeric_columns].min())
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()

def clean_dataset(df, missing_strategy='mean', outlier_method='zscore', standardize=False):
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if outlier_method == 'zscore':
        cleaner.remove_outliers_zscore()
    elif outlier_method == 'iqr':
        cleaner.remove_outliers_iqr()
    
    if standardize:
        cleaner.standardize_data()
    
    return cleaner.get_cleaned_data()
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in [np.float64, np.int64]:
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
            elif cleaned_df[column].dtype == 'object':
                cleaned_df[column] = cleaned_df[column].fillna('Unknown')
        print("Missing values have been filled")
    
    return cleaned_df

def validate_dataset(df):
    """
    Validate the dataset for common issues.
    """
    issues = []
    
    if df.isnull().sum().sum() > 0:
        issues.append(f"Dataset contains {df.isnull().sum().sum()} missing values")
    
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Dataset contains {duplicate_count} duplicate rows")
    
    for column in df.columns:
        if df[column].nunique() == 1:
            issues.append(f"Column '{column}' has only one unique value")
    
    if issues:
        print("Validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve'],
        'age': [25, 30, None, 35, 40, 40],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    is_valid = validate_dataset(cleaned_df)
    print(f"\nDataset validation passed: {is_valid}")