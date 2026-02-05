import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicate rows and
    filling missing values with appropriate defaults.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing values
    # For numeric columns, fill with median
    # For categorical columns, fill with mode
    for column in df_cleaned.columns:
        if pd.api.types.is_numeric_dtype(df_cleaned[column]):
            median_value = df_cleaned[column].median()
            df_cleaned[column] = df_cleaned[column].fillna(median_value)
        else:
            mode_value = df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else 'Unknown'
            df_cleaned[column] = df_cleaned[column].fillna(mode_value)
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_data(df, required_columns=None):
    """
    Validate that the DataFrame meets basic quality requirements.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = pd.DataFrame({
#         'A': [1, 2, None, 4, 1],
#         'B': ['x', 'y', None, 'x', 'z'],
#         'C': [10.5, 20.3, 15.7, None, 10.5]
#     })
#     
#     cleaned_data = clean_dataset(sample_data)
#     print("Original data:")
#     print(sample_data)
#     print("\nCleaned data:")
#     print(cleaned_data)import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        return clean_df
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
                clean_df = clean_df[(z_scores < threshold) | clean_df[col].isna()]
        return clean_df
    
    def impute_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        imputed_df = self.df.copy()
        for col in columns:
            if col in imputed_df.columns:
                mean_val = imputed_df[col].mean()
                imputed_df[col] = imputed_df[col].fillna(mean_val)
        return imputed_df
    
    def impute_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        imputed_df = self.df.copy()
        for col in columns:
            if col in imputed_df.columns:
                median_val = imputed_df[col].median()
                imputed_df[col] = imputed_df[col].fillna(median_val)
        return imputed_df
    
    def drop_missing_columns(self, threshold=0.3):
        missing_ratio = self.df.isnull().sum() / len(self.df)
        columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        return self.df.drop(columns=columns_to_drop)
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': self.numeric_columns,
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(100, 10), 'feature_a'] = np.nan
    df.loc[np.random.choice(100, 5), 'feature_b'] = np.nan
    df.loc[95:99, 'feature_a'] = 500
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Data Summary:")
    print(f"Original shape: {cleaner.get_summary()['original_shape']}")
    print(f"Missing values: {cleaner.get_summary()['missing_values']}")
    
    cleaned_df = cleaner.remove_outliers_iqr(['feature_a'])
    imputed_df = cleaner.impute_missing_mean()
    
    print(f"\nAfter outlier removal: {cleaned_df.shape}")
    print(f"After imputation: {imputed_df.isnull().sum().sum()} missing values remaining")