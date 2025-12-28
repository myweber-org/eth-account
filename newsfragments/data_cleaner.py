
import numpy as np
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
            if col in self.numeric_columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                mask = (clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)
                clean_df = clean_df[mask]
        
        return clean_df
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
                mask = z_scores < threshold
                clean_df = clean_df[mask]
        
        return clean_df
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        
        return normalized_df
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val > 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        
        return normalized_df
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                filled_df[col] = filled_df[col].fillna(filled_df[col].mean())
        
        return filled_df
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                filled_df[col] = filled_df[col].fillna(filled_df[col].median())
        
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df.isnull().sum().to_dict(),
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
    df.loc[np.random.choice(100, 5), 'feature_a'] = np.nan
    df.loc[np.random.choice(100, 3), 'feature_b'] = np.nan
    
    outliers = np.random.choice(100, 5)
    df.loc[outliers, 'feature_a'] = df['feature_a'].mean() + 5 * df['feature_a'].std()
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Data Summary:")
    summary = cleaner.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nCleaning with IQR method...")
    clean_iqr = cleaner.remove_outliers_iqr()
    print(f"Original shape: {sample_df.shape}, Cleaned shape: {clean_iqr.shape}")
    
    print("\nNormalizing data...")
    normalized = cleaner.normalize_minmax()
    print("Normalized data sample:")
    print(normalized.head())
    
    print("\nFilling missing values...")
    filled = cleaner.fill_missing_mean()
    print("Missing values after filling:")
    print(filled.isnull().sum())import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        Cleaned DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in DataFrame columns.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: list of columns to fill (None for all numeric columns)
    
    Returns:
        DataFrame with missing values filled
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to normalize (None for all numeric columns)
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def clean_dataframe(df, remove_dups=True, fill_na=True, normalize=True):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df: pandas DataFrame
        remove_dups: whether to remove duplicates
        fill_na: whether to fill missing values
        normalize: whether to normalize numeric columns
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df)
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df)
    
    return cleaned_dfimport pandas as pd
import numpy as np

def load_data(filepath):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def remove_outliers(df, column, threshold=3):
    """Remove outliers using Z-score method."""
    if column not in df.columns:
        print(f"Column '{column}' not found in dataframe.")
        return df
    
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    filtered_df = df[z_scores < threshold]
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'.")
    return filtered_df

def normalize_column(df, column):
    """Normalize column values to range [0,1]."""
    if column not in df.columns:
        print(f"Column '{column}' not found in dataframe.")
        return df
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        print(f"Column '{column}' has constant values. Skipping normalization.")
        return df
    
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    print(f"Column '{column}' normalized successfully.")
    return df

def clean_data(df, numeric_columns):
    """Main data cleaning pipeline."""
    if df is None or df.empty:
        print("No data to clean.")
        return df
    
    print("Starting data cleaning process...")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_count - len(df)
    print(f"Removed {duplicates_removed} duplicate rows.")
    
    # Handle missing values
    for col in numeric_columns:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"Filled {missing_count} missing values in column '{col}' with median.")
    
    # Remove outliers from numeric columns
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers(df, col)
    
    # Normalize numeric columns
    for col in numeric_columns:
        if col in df.columns:
            df = normalize_column(df, col)
    
    print("Data cleaning completed.")
    return df

def save_cleaned_data(df, output_path):
    """Save cleaned data to CSV file."""
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to '{output_path}'")
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    # Define numeric columns for processing
    numeric_cols = ['age', 'income', 'score']
    
    # Load data
    data = load_data(input_file)
    
    if data is not None:
        # Clean data
        cleaned_data = clean_data(data, numeric_cols)
        
        # Save cleaned data
        save_cleaned_data(cleaned_data, output_file)