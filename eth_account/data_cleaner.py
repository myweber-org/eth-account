
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean.reset_index(drop=True)
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_standardized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_standardized[col] = (self.df[col] - mean_val) / std_val
        
        self.df = df_standardized
        return self
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    continue
                
                df_filled[col] = self.df[col].fillna(fill_value)
        
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0]
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'feature3': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'feature1'] = np.nan
    df.loc[5, 'feature2'] = 1000
    
    cleaner = DataCleaner(df)
    print(f"Original shape: {cleaner.original_shape}")
    
    removed = cleaner.remove_outliers_iqr(['feature2'])
    print(f"Removed {removed} outliers")
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.normalize_minmax()
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Summary: {summary}")
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(result.head())
import pandas as pd
import numpy as np
from typing import Optional, List

def clean_csv_data(
    input_path: str,
    output_path: str,
    columns_to_drop: Optional[List[str]] = None,
    fill_strategy: str = 'mean',
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and removing unnecessary columns.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        columns_to_drop: List of column names to drop
        fill_strategy: Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
        threshold: Maximum allowed missing value ratio per column (0-1)
    
    Returns:
        Cleaned DataFrame
    """
    
    df = pd.read_csv(input_path)
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"Dropped columns: {columns_to_drop}")
    
    missing_ratios = df.isnull().sum() / len(df)
    columns_to_remove = missing_ratios[missing_ratios > threshold].index.tolist()
    
    if columns_to_remove:
        df = df.drop(columns=columns_to_remove)
        print(f"Removed columns with >{threshold*100}% missing values: {columns_to_remove}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if fill_strategy == 'mean':
                fill_value = df[col].mean()
            elif fill_strategy == 'median':
                fill_value = df[col].median()
            elif fill_strategy == 'zero':
                fill_value = 0
            else:
                fill_value = df[col].mean()
            
            df[col] = df[col].fillna(fill_value)
            print(f"Filled missing values in '{col}' with {fill_strategy}: {fill_value}")
    
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_value)
            print(f"Filled missing values in '{col}' with mode: {mode_value}")
    
    df.to_csv(output_path, index=False)
    
    final_shape = df.shape
    print(f"Cleaned data shape: {final_shape}")
    print(f"Removed {original_shape[1] - final_shape[1]} columns")
    print(f"Cleaned data saved to: {output_path}")
    
    return df

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame for common data quality issues.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if data passes validation checks
    """
    checks = []
    
    checks.append(('No null values', df.isnull().sum().sum() == 0))
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].nunique() == 1:
            checks.append((f'Column {col} has only one unique value', False))
    
    if len(df) == 0:
        checks.append(('DataFrame is empty', False))
    
    all_passed = all(check[1] for check in checks)
    
    if not all_passed:
        failed_checks = [check[0] for check in checks if not check[1]]
        print(f"Validation failed for: {failed_checks}")
    
    return all_passed

if __name__ == "__main__":
    cleaned_df = clean_csv_data(
        input_path='raw_data.csv',
        output_path='cleaned_data.csv',
        columns_to_drop=['id', 'timestamp'],
        fill_strategy='median',
        threshold=0.3
    )
    
    is_valid = validate_dataframe(cleaned_df)
    print(f"Data validation: {'PASSED' if is_valid else 'FAILED'}")