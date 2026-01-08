import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names (if None, auto-detect)
        outlier_factor: IQR multiplier for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            # Remove outliers
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_factor)
            
            # Normalize the column
            cleaned_data[f"{column}_normalized"] = normalize_minmax(cleaned_data, column)
            cleaned_data[f"{column}_standardized"] = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate data structure and content.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        allow_nan: whether to allow NaN values
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if not allow_nan and data.isnull().any().any():
        nan_cols = data.columns[data.isnull().any()].tolist()
        return False, f"NaN values found in columns: {nan_cols}"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    
    # Clean the data
    cleaned = clean_dataset(sample_data, numeric_columns=['feature1', 'feature2'])
    print("Cleaned data shape:", cleaned.shape)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, allow_nan=False)
    print(f"Validation: {is_valid}, Message: {message}")
import pandas as pd
import hashlib

def remove_duplicates_by_hash(df, columns):
    """
    Remove duplicate rows based on a hash of specified columns.
    """
    if not columns:
        raise ValueError("Columns list cannot be empty")
    
    df['_hash'] = df[columns].apply(
        lambda row: hashlib.md5(
            ''.join(row.astype(str)).encode()
        ).hexdigest(),
        axis=1
    )
    
    df_cleaned = df.drop_duplicates(subset=['_hash'])
    df_cleaned = df_cleaned.drop(columns=['_hash'])
    
    return df_cleaned.reset_index(drop=True)

def clean_numeric_columns(df, columns, fill_method='mean'):
    """
    Clean numeric columns by filling missing values.
    """
    for col in columns:
        if col not in df.columns:
            continue
            
        if fill_method == 'mean':
            fill_value = df[col].mean()
        elif fill_method == 'median':
            fill_value = df[col].median()
        elif fill_method == 'mode':
            fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
        else:
            fill_value = 0
        
        df[col] = df[col].fillna(fill_value)
    
    return df

def main():
    # Example usage
    data = {
        'id': [1, 2, 3, 4, 5, 3],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Charlie'],
        'age': [25, 30, None, 35, 40, 35],
        'score': [85.5, 92.0, 78.5, None, 88.0, 78.5]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Remove duplicates
    df_unique = remove_duplicates_by_hash(df, ['name', 'age', 'score'])
    print("After removing duplicates:")
    print(df_unique)
    print()
    
    # Clean numeric columns
    df_cleaned = clean_numeric_columns(df_unique, ['age', 'score'], fill_method='mean')
    print("After cleaning numeric columns:")
    print(df_cleaned)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not process column '{col}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['X', 'Y', 'Z'], 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original shape: {df.shape}")
    
    cleaned_df = clean_dataset(df)
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Rows removed: {len(df) - len(cleaned_df)}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df[column])
    }
    
    return stats

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 200],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nDataset shape:", df.shape)
    
    cleaned_df = clean_dataset(df, ['temperature', 'humidity', 'pressure'])
    print("\nCleaned dataset:")
    print(cleaned_df)
    print("\nCleaned dataset shape:", cleaned_df.shape)
    
    for column in ['temperature', 'humidity', 'pressure']:
        stats = calculate_summary_statistics(cleaned_df, column)
        print(f"\nStatistics for {column}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
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
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
        print(f"Filled missing numeric values with {fill_missing}.")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
        print("Filled missing categorical values with mode.")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("Warning: DataFrame is empty.")
    
    return True

def get_data_summary(df):
    """
    Generate summary statistics for the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    dict: Dictionary containing summary statistics.
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'numeric_stats': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {}
    }
    return summary

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None],
        'B': [10, 20, 20, 40, 50, 60],
        'C': ['x', 'y', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    print("\nData Summary:")
    summary = get_data_summary(cleaned)
    for key, value in summary.items():
        print(f"{key}: {value}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Strategy to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {original_shape[0] - cleaned_df.shape[0]} duplicate rows")
    
    # Handle missing values
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values")
        elif fill_missing == 'mean':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            print("Filled missing values with column means")
        elif fill_missing == 'median':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            print("Filled missing values with column medians")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].isnull().any():
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
            print("Filled missing values with column modes")
    
    # Report cleaning summary
    final_shape = cleaned_df.shape
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {final_shape}")
    print(f"Rows removed: {original_shape[0] - final_shape[0]}")
    print(f"Columns: {original_shape[1]}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Error: DataFrame is None or empty")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

# Example usage function
def process_sample_data():
    """Demonstrate the data cleaning functionality with sample data."""
    # Create sample data with issues
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10.5, np.nan, 15.2, 15.2, np.nan, 20.1],
        'category': ['A', 'B', 'A', 'A', 'B', 'C'],
        'score': [85, 92, np.nan, 78, 88, 95]
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid = validate_dataframe(cleaned, required_columns=['id', 'value'], min_rows=3)
    print(f"\nData validation passed: {is_valid}")
    
    return cleaned

if __name__ == "__main__":
    # Run the sample demonstration
    result = process_sample_data()
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, fill_method='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path: Path to input CSV file
        output_path: Path for cleaned CSV output (optional)
        fill_method: Method for filling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
        DataFrame containing cleaned data
    """
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        original_rows = len(df)
        
        df = df.drop_duplicates()
        print(f"Removed {original_rows - len(df)} duplicate rows")
        
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Missing values found:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count} missing")
            
            if fill_method == 'mean':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_method == 'median':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_method == 'mode':
                for col in df.columns:
                    if df[col].isnull().any():
                        mode_val = df[col].mode()
                        if not mode_val.empty:
                            df[col] = df[col].fillna(mode_val[0])
            elif fill_method == 'drop':
                df = df.dropna()
                print(f"Dropped rows with missing values")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        print(f"Final data shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
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
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', None, 'A', 'C', 'A']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv', fill_method='mean')
    
    if cleaned_df is not None:
        validation_result = validate_dataframe(cleaned_df, ['id', 'value', 'category'])
        print(f"Validation result: {validation_result}")