import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr', normalize=False):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    if normalize:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = normalize_minmax(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in dataset")
    
    return True

def main():
    sample_data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original dataset shape: {df.shape}")
    
    numeric_cols = ['feature1', 'feature2', 'feature3']
    
    try:
        validate_data(df, numeric_cols)
        cleaned_df = clean_dataset(df, numeric_cols, method='iqr', normalize=True)
        print(f"Cleaned dataset shape: {cleaned_df.shape}")
        print(f"Outliers removed: {len(df) - len(cleaned_df)}")
        
        print("\nSummary statistics:")
        print(cleaned_df[numeric_cols].describe())
        
    except ValueError as e:
        print(f"Data validation error: {e}")

if __name__ == "__main__":
    main()import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', output_path=None):
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        filepath: Path to the CSV file
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
        output_path: Optional path to save cleaned data
    
    Returns:
        Cleaned pandas DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            if missing_strategy == 'mean':
                df = df.fillna(df.mean(numeric_only=True))
            elif missing_strategy == 'median':
                df = df.fillna(df.median(numeric_only=True))
            elif missing_strategy == 'drop':
                df = df.dropna()
            elif missing_strategy == 'zero':
                df = df.fillna(0)
            else:
                raise ValueError(f"Unknown strategy: {missing_strategy}")
            
            print(f"Missing values handled using '{missing_strategy}' strategy")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        if removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows")
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        # Save cleaned data if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Boolean indicating if validation passed
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Validation failed: Missing columns {missing_columns}")
            return False
    
    print("Data validation passed")
    return True

# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, 7, 8, 9],
        'C': [10, 11, 12, np.nan, 14]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 
                               missing_strategy='mean',
                               output_path='cleaned_data.csv')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
        print(f"DataFrame valid: {is_valid}")
        print(f"Cleaned DataFrame shape: {cleaned_df.shape}")
        print(cleaned_df.head())
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            original_count = len(df)
            df = remove_outliers_iqr(df, column)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column: {column}")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved to: {output_file}")
        print(f"Final dataset shape: {df.shape}")
        
        return True
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = original_shape[0] - cleaned_df.shape[0]
        print(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print(f"Dropped rows with missing values")
        elif fill_missing == 'mean':
            for column in cleaned_df.select_dtypes(include=[np.number]).columns:
                if cleaned_df[column].isnull().any():
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            print("Filled missing values with column means")
        elif fill_missing == 'median':
            for column in cleaned_df.select_dtypes(include=[np.number]).columns:
                if cleaned_df[column].isnull().any():
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            print("Filled missing values with column medians")
        elif fill_missing == 'mode':
            for column in cleaned_df.columns:
                if cleaned_df[column].isnull().any():
                    if cleaned_df[column].dtype == 'object':
                        mode_value = cleaned_df[column].mode()[0]
                        cleaned_df[column].fillna(mode_value, inplace=True)
                    else:
                        cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            print("Filled missing values with column modes/medians")
    
    # Report cleaning statistics
    final_shape = cleaned_df.shape
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {final_shape}")
    print(f"Rows removed: {original_shape[0] - final_shape[0]}")
    print(f"Columns: {original_shape[1]} (unchanged)")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("DataFrame is None or empty")
        return False
    
    if len(df) < min_rows:
        print(f"DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    sample_data = {
        'id': [1, 2, 3, 1, 2, 6, 7],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'Eve', None],
        'age': [25, 30, None, 25, 30, 28, 35],
        'score': [85.5, 92.0, 78.5, 85.5, 92.0, 88.0, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    print("\nValidation result:", validate_dataframe(cleaned, required_columns=['id', 'name'], min_rows=1))