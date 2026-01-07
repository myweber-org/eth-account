import pandas as pd
import numpy as np

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

def clean_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_cols:
        df = normalize_minmax(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Original shape: {pd.read_csv(input_path).shape}")
    print(f"Cleaned shape: {df.shape}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'constant':
            if fill_value is not None:
                self.df.fillna(fill_value, inplace=True)
            else:
                raise ValueError("fill_value must be provided for constant strategy")
        return self
    
    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
            self.df = self.df[(z_scores < threshold).all(axis=1)]
        elif method == 'iqr':
            for col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self
    
    def normalize_data(self, method='minmax'):
        if method == 'minmax':
            for col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val != min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        elif method == 'standard':
            for col in self.numeric_columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val != 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def summary(self):
        print("Data Summary:")
        print(f"Shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        print(f"Numeric columns: {len(self.numeric_columns)}")
        print(f"Categorical columns: {len(self.categorical_columns)}")
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list): List of column names to check for duplicates.
                             If None, uses all columns.
    fill_strategy (str): Strategy to fill missing values.
                         Options: 'mean', 'median', 'mode', 'drop', 'zero'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    original_shape = df.shape
    
    # Remove duplicate rows
    if columns_to_check is None:
        df_cleaned = df.drop_duplicates()
    else:
        df_cleaned = df.drop_duplicates(subset=columns_to_check)
    
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    
    # Handle missing values
    missing_before = df_cleaned.isnull().sum().sum()
    
    if fill_strategy == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif fill_strategy in ['mean', 'median']:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_strategy == 'mean':
                fill_value = df_cleaned[col].mean()
            else:  # median
                fill_value = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    elif fill_strategy == 'mode':
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                mode_value = df_cleaned[col].mode()
                if not mode_value.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value.iloc[0])
    elif fill_strategy == 'zero':
        df_cleaned = df_cleaned.fillna(0)
    
    missing_after = df_cleaned.isnull().sum().sum()
    
    # Print cleaning summary
    print(f"Original dataset shape: {original_shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Missing values before cleaning: {missing_before}")
    print(f"Missing values after cleaning: {missing_after}")
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    
    return df_cleaned

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the cleaned DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: Less than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing columns: {missing_cols}")
            return False
    
    # Check for remaining missing values
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Validation warning: {remaining_missing} missing values remain")
    
    print("Data validation passed")
    return True

# Example usage function
def process_sample_data():
    """
    Create and process sample data to demonstrate the cleaning functions.
    """
    # Create sample data with duplicates and missing values
    np.random.seed(42)
    sample_data = {
        'id': [1, 2, 3, 1, 2, 6, 7, 8],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'Eve', 'Frank', 'Grace'],
        'age': [25, 30, 35, 25, 30, 28, np.nan, 32],
        'score': [85.5, 92.0, 78.5, 85.5, 92.0, 88.0, 76.5, np.nan],
        'department': ['HR', 'IT', 'Sales', 'HR', 'IT', 'IT', 'Sales', 'HR']
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample data created")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'], fill_strategy='mean')
    
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned data
    required_cols = ['id', 'name', 'age', 'score']
    is_valid = validate_data(cleaned_df, required_columns=required_cols, min_rows=5)
    
    if is_valid:
        print("\nCleaned data is ready for analysis:")
        print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    # Run the sample data processing
    result_df = process_sample_data()