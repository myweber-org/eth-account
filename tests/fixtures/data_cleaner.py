
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str): Method to fill missing values: 'mean', 'median', 'mode', or 'drop'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'])
    print(f"\nData validation passed: {is_valid}")
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_data(df, columns, method='minmax'):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            if method == 'minmax':
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val > 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    return normalized_df

def handle_missing_values(df, strategy='mean'):
    processed_df = df.copy()
    for col in processed_df.columns:
        if processed_df[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(processed_df[col]):
                processed_df[col].fillna(processed_df[col].mean(), inplace=True)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(processed_df[col]):
                processed_df[col].fillna(processed_df[col].median(), inplace=True)
            elif strategy == 'mode':
                processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
            else:
                processed_df[col].fillna(0, inplace=True)
    return processed_df

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    df_processed = handle_missing_values(df, strategy=missing_strategy)
    
    if outlier_method == 'iqr':
        df_processed = remove_outliers_iqr(df_processed, numeric_columns)
    elif outlier_method == 'zscore':
        z_scores = np.abs(stats.zscore(df_processed[numeric_columns]))
        df_processed = df_processed[(z_scores < 3).all(axis=1)]
    
    df_processed = normalize_data(df_processed, numeric_columns, method=normalize_method)
    
    return df_processed.reset_index(drop=True)import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, fill_strategy='mean', drop_threshold=0.5):
    """
    Clean CSV data by handling missing values and removing low-quality columns.
    
    Parameters:
    file_path (str): Path to input CSV file.
    output_path (str, optional): Path for cleaned CSV output. If None, returns DataFrame.
    fill_strategy (str): Method for filling missing values ('mean', 'median', 'mode', 'zero').
    drop_threshold (float): Drop columns with missing ratio above this threshold (0.0-1.0).
    
    Returns:
    pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None.
    """
    
    df = pd.read_csv(file_path)
    original_shape = df.shape
    
    missing_ratios = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratios[missing_ratios > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            if fill_strategy == 'mean':
                fill_value = df[column].mean()
            elif fill_strategy == 'median':
                fill_value = df[column].median()
            elif fill_strategy == 'zero':
                fill_value = 0
            else:
                fill_value = df[column].mode()[0] if not df[column].mode().empty else 0
        else:
            fill_value = df[column].mode()[0] if not df[column].mode().empty else 'Unknown'
        
        df[column] = df[column].fillna(fill_value)
    
    df = df.drop_duplicates()
    
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df.shape}")
    print(f"Dropped columns: {list(columns_to_drop)}")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return None
    else:
        return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of columns that must be present.
    
    Returns:
    dict: Validation results.
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].abs().max() > 1e10:
            validation_results['warnings'].append(f"Column '{col}' contains extremely large values")
    
    return validation_results

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, np.nan, np.nan, 10, 11],
        'C': [7, 8, 9, 10, 11],
        'D': ['x', 'y', np.nan, 'z', 'x']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='median', drop_threshold=0.6)
    
    validation = validate_dataframe(cleaned_df, required_columns=['A', 'C'])
    print(f"Validation passed: {validation['is_valid']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            df = remove_outliers_iqr(df, col)
        
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Error cleaning dataset: {e}")
        return None

if __name__ == "__main__":
    cleaned_data = clean_dataset("raw_data.csv")
    if cleaned_data is not None:
        cleaned_data.to_csv("cleaned_data.csv", index=False)
        print("Data cleaning completed successfully.")
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Load a CSV file, clean missing values, convert data types,
    and save the cleaned version.
    """
    df = pd.read_csv(input_path)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Fill missing numeric values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # Convert date columns if present
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except:
            pass
    
    # Remove rows where critical columns are still null
    critical_cols = [col for col in df.columns if 'id' in col.lower() or 'key' in col.lower()]
    if critical_cols:
        df = df.dropna(subset=critical_cols)
    
    # Generate output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    print(f"Original shape: {pd.read_csv(input_path).shape}, Cleaned shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'user_id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'age': [25, 30, None, 35, 40, 40],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0],
        'join_date': ['2023-01-15', '2023-02-20', None, '2023-03-10', '2023-04-05', '2023-04-05']
    }
    
    # Create test CSV
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data('test_data.csv')
    
    # Display results
    print("\nFirst few rows of cleaned data:")
    print(cleaned_df.head())