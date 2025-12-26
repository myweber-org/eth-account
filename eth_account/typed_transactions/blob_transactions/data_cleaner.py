import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def process_data_file(file_path, output_path=None):
    """
    Load, clean, and save a data file.
    
    Args:
        file_path (str): Path to input data file
        output_path (str): Path to save cleaned data (optional)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")
        
        print(f"Loaded data with shape: {df.shape}")
        
        if validate_dataframe(df):
            cleaned_df = clean_dataframe(df)
            print(f"Cleaned data shape: {cleaned_df.shape}")
            
            if output_path:
                if output_path.endswith('.csv'):
                    cleaned_df.to_csv(output_path, index=False)
                elif output_path.endswith('.xlsx'):
                    cleaned_df.to_excel(output_path, index=False)
                print(f"Saved cleaned data to: {output_path}")
            
            return cleaned_df
        else:
            print("Data validation failed")
            return None
            
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'value': [10, 20, 20, np.nan, 40],
        'category': ['A', 'B', 'B', 'C', None]
    })
    
    print("Original data:")
    print(sample_data)
    print("\nCleaning data...")
    
    cleaned = clean_dataframe(sample_data)
    print("\nCleaned data:")
    print(cleaned)
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
        method (str): Outlier detection method ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        mask = z_scores <= threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Clean CSV data by handling missing values and removing low-quality columns.
    
    Parameters:
    filepath (str): Path to the CSV file.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero').
    drop_threshold (float): Drop columns with missing ratio above this threshold.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    # Calculate missing ratio per column
    missing_ratio = df.isnull().sum() / len(df)
    
    # Drop columns with high missing ratio
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    if len(columns_to_drop) > 0:
        print(f"Dropped columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
    
    # Fill remaining missing values based on strategy
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'mean':
        fill_values = df[numeric_cols].mean()
    elif fill_strategy == 'median':
        fill_values = df[numeric_cols].median()
    elif fill_strategy == 'mode':
        fill_values = df[numeric_cols].mode().iloc[0]
    elif fill_strategy == 'zero':
        fill_values = 0
    else:
        raise ValueError(f"Unknown fill strategy: {fill_strategy}")
    
    df[numeric_cols] = df[numeric_cols].fillna(fill_values)
    
    # For categorical columns, fill with most frequent value
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        most_frequent = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col] = df[col].fillna(most_frequent)
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Removed {original_shape[1] - df.shape[1]} columns")
    
    return df

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to check for outliers.
    multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
    pd.Series: Boolean series indicating outliers.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not np.issubdtype(df[column].dtype, np.number):
        raise ValueError(f"Column '{column}' must be numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    outlier_count = outliers.sum()
    if outlier_count > 0:
        print(f"Detected {outlier_count} outliers in column '{column}'")
    
    return outliers

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame.
    output_path (str): Path to save the cleaned data.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [np.nan, np.nan, 3, 4, 5, 6],
        'C': ['a', 'b', np.nan, 'a', 'b', 'c'],
        'D': [1, 2, 3, 4, 5, 6]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', fill_strategy='median', drop_threshold=0.3)
    
    outliers = detect_outliers_iqr(cleaned_df, 'A')
    print(f"Outlier indices: {cleaned_df[outliers].index.tolist()}")
    
    save_cleaned_data(cleaned_df, 'cleaned_test_data.csv')