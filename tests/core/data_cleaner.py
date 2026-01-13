
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or array-like): The dataset
    column (int or str): Column index or name if using pandas DataFrame
    
    Returns:
    tuple: (cleaned_data, outliers_removed)
    """
    if isinstance(data, list):
        data_array = np.array(data)
    else:
        data_array = data
    
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (data_array >= lower_bound) & (data_array <= upper_bound)
    cleaned_data = data_array[mask]
    outliers = data_array[~mask]
    
    return cleaned_data, outliers

def calculate_statistics(data):
    """
    Calculate basic statistics for the dataset.
    
    Parameters:
    data (array-like): Input data
    
    Returns:
    dict: Dictionary containing statistics
    """
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'count': len(data)
    }
    return stats

def clean_dataset(data, columns=None):
    """
    Clean dataset by removing outliers from specified columns.
    
    Parameters:
    data (array-like or dict): Input data
    columns (list): List of columns to clean
    
    Returns:
    dict: Dictionary with cleaned data and statistics
    """
    if columns is None:
        if isinstance(data, dict):
            columns = list(data.keys())
        else:
            columns = [0]
    
    result = {}
    
    for col in columns:
        if isinstance(data, dict):
            col_data = data[col]
        else:
            col_data = data[:, col] if hasattr(data, 'shape') else data
        
        cleaned, outliers = remove_outliers_iqr(col_data, col)
        stats = calculate_statistics(cleaned)
        
        result[col] = {
            'cleaned_data': cleaned,
            'outliers': outliers,
            'statistics': stats,
            'outliers_count': len(outliers)
        }
    
    return result

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.normal(100, 15, 1000)
    sample_data_with_outliers = np.append(sample_data, [10, 200, 300, -50])
    
    cleaned, outliers = remove_outliers_iqr(sample_data_with_outliers, 0)
    print(f"Original data points: {len(sample_data_with_outliers)}")
    print(f"Cleaned data points: {len(cleaned)}")
    print(f"Outliers removed: {len(outliers)}")
    
    stats = calculate_statistics(cleaned)
    print(f"Statistics after cleaning: {stats}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Strategy to fill missing values ('mean', 'median', 'mode', or 'drop').
    
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
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}.")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
        print("Filled missing values with mode.")
    
    return cleaned_df

def validate_dataset(df, check_duplicates=True, check_missing=True):
    """
    Validate a DataFrame for duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    check_duplicates (bool): Check for duplicate rows.
    check_missing (bool): Check for missing values.
    
    Returns:
    dict: Validation results.
    """
    validation_results = {}
    
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        validation_results['duplicates'] = duplicate_count
    
    if check_missing:
        missing_count = df.isnull().sum().sum()
        validation_results['missing_values'] = missing_count
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None],
        'B': [10, None, 30, 40, 50, 60],
        'C': ['x', 'y', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    validation = validate_dataset(df)
    print(f"\nValidation Results: {validation}")
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list, optional): List of column names to check for duplicates.
                                       If None, checks all columns.
    fill_missing (str or dict, optional): Method to fill missing values.
                                          Can be 'mean', 'median', 'mode', or a dictionary
                                          of column:value pairs.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    original_shape = df.shape
    
    # Remove duplicate rows
    if columns_to_check is None:
        df_clean = df.drop_duplicates()
    else:
        df_clean = df.drop_duplicates(subset=columns_to_check)
    
    # Handle missing values
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            df_clean = df_clean.fillna(fill_missing)
        elif fill_missing == 'mean':
            numeric_cols = df_clean.select_dtypes(include=['number']).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif fill_missing == 'median':
            numeric_cols = df_clean.select_dtypes(include=['number']).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif fill_missing == 'mode':
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category':
                    mode_val = df_clean[col].mode()
                    if not mode_val.empty:
                        df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    # Print cleaning summary
    duplicates_removed = original_shape[0] - df_clean.shape[0]
    missing_filled = df.isna().sum().sum() - df_clean.isna().sum().sum()
    
    print(f"Original dataset shape: {original_shape}")
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Missing values filled: {missing_filled}")
    print(f"Remaining missing values: {df_clean.isna().sum().sum()}")
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Error: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns is not None:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
#         'age': [25, 30, 30, None, 35, 40],
#         'score': [85.5, 92.0, 92.0, 78.5, None, 95.0]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\n" + "="*50 + "\n")
#     
#     # Clean the data
#     cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'], fill_missing='mean')
#     
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)