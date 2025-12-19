import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values
                               Options: 'mean', 'median', 'drop', 'zero'
    """
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        df_cleaned = df.copy()
        
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"After removing duplicates: {df_cleaned.shape}")
        
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
        elif missing_strategy == 'median':
            for col in numeric_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        elif missing_strategy == 'zero':
            df_cleaned.fillna(0, inplace=True)
        elif missing_strategy == 'drop':
            df_cleaned.dropna(inplace=True)
        
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_cleaned[col].isnull().any():
                df_cleaned[col].fillna('Unknown', inplace=True)
        
        print(f"Final cleaned data shape: {df_cleaned.shape}")
        print(f"Missing values after cleaning:\n{df_cleaned.isnull().sum().sum()}")
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df, required_columns=None):
    """
    Validate cleaned data for basic quality checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': []
    }
    
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty or None')
        return validation_results
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing required columns: {missing_cols}')
    
    if df.isnull().sum().sum() > 0:
        validation_results['is_valid'] = False
        validation_results['issues'].append('Data contains missing values')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].min() < 0 and col not in ['temperature', 'balance']:
            validation_results['issues'].append(f'Column {col} contains negative values')
    
    return validation_results

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_data = clean_csv_data(input_csv, output_csv, missing_strategy='mean')
    
    if cleaned_data is not None:
        validation = validate_data(cleaned_data)
        print(f"Data validation passed: {validation['is_valid']}")
        if not validation['is_valid']:
            print(f"Validation issues: {validation['issues']}")
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        if self.file_path.exists():
            self.df = pd.read_csv(self.file_path)
            return True
        return False
    
    def remove_duplicates(self):
        if self.df is not None:
            initial_rows = len(self.df)
            self.df = self.df.drop_duplicates()
            removed = initial_rows - len(self.df)
            return removed
        return 0
    
    def fill_missing_values(self, strategy='mean', fill_value=None):
        if self.df is not None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            if strategy == 'mean':
                for col in numeric_cols:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == 'median':
                for col in numeric_cols:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == 'custom' and fill_value is not None:
                self.df = self.df.fillna(fill_value)
            
            return len(numeric_cols)
        return 0
    
    def remove_outliers(self, threshold=3):
        if self.df is not None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            initial_rows = len(self.df)
            
            for col in numeric_cols:
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < threshold]
            
            removed = initial_rows - len(self.df)
            return removed
        return 0
    
    def save_cleaned_data(self, output_path=None):
        if self.df is not None:
            if output_path is None:
                output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
            
            self.df.to_csv(output_path, index=False)
            return str(output_path)
        return None
    
    def get_summary(self):
        if self.df is not None:
            summary = {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'missing_values': self.df.isnull().sum().sum(),
                'data_types': self.df.dtypes.to_dict()
            }
            return summary
        return None

def process_csv_file(input_file, output_dir='cleaned_data'):
    cleaner = DataCleaner(input_file)
    
    if not cleaner.load_data():
        print(f"Error: File {input_file} not found")
        return None
    
    print(f"Loaded data: {len(cleaner.df)} rows, {len(cleaner.df.columns)} columns")
    
    duplicates_removed = cleaner.remove_duplicates()
    print(f"Removed {duplicates_removed} duplicate rows")
    
    cols_filled = cleaner.fill_missing_values(strategy='mean')
    print(f"Filled missing values in {cols_filled} numeric columns")
    
    outliers_removed = cleaner.remove_outliers()
    print(f"Removed {outliers_removed} outlier rows")
    
    output_path = cleaner.save_cleaned_data()
    print(f"Saved cleaned data to: {output_path}")
    
    summary = cleaner.get_summary()
    print(f"Final dataset: {summary['rows']} rows, {summary['columns']} columns")
    
    return output_path
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_clean (list, optional): List of column names to apply string normalization.
                                       If None, applies to all object dtype columns.
    remove_duplicates (bool): If True, remove duplicate rows.
    case_normalization (str): One of 'lower', 'upper', or None for case normalization.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if columns_to_clean is None:
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_clean:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].astype(str)
            
            if case_normalization == 'lower':
                cleaned_df[col] = cleaned_df[col].str.lower()
            elif case_normalization == 'upper':
                cleaned_df[col] = cleaned_df[col].str.upper()
            
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    email_column (str): Name of the column containing email addresses.
    
    Returns:
    pd.DataFrame: DataFrame with additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].str.match(email_pattern, na=False)
    
    valid_count = df['email_valid'].sum()
    print(f"Found {valid_count} valid email addresses out of {len(df)} rows.")
    
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'ALICE BROWN', '  Bob White  '],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'alice@company.co.uk', 'bob@domain.com'],
        'age': [25, 30, 25, 35, 40]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataframe(df, columns_to_clean=['name'], remove_duplicates=True, case_normalization='lower')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    validated = validate_email_column(cleaned, 'email')
    print("DataFrame with email validation:")
    print(validated)
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Strategy for filling missing values: 'mean', 'median', 'mode', or 'drop'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                cleaned_df[column].fillna(fill_value, inplace=True)
                print(f"Filled missing values in column '{column}' with {fill_missing}: {fill_value:.2f}")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if len(df) < min_rows:
        print(f"DataFrame has only {len(df)} rows, minimum required is {min_rows}.")
        return False
    
    if df.isnull().all().any():
        print("Some columns contain only missing values.")
        return False
    
    print("Data validation passed.")
    return True

def main():
    sample_data = {
        'A': [1, 2, 2, 3, np.nan],
        'B': [4, np.nan, 6, 6, 8],
        'C': [7, 8, 9, 9, 10]
    }
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\nCleaning data...")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    print("\nValidating cleaned data...")
    is_valid = validate_data(cleaned, required_columns=['A', 'B', 'C'], min_rows=2)
    
    if is_valid:
        print("Data cleaning completed successfully.")
    else:
        print("Data cleaning completed with validation issues.")

if __name__ == "__main__":
    main()