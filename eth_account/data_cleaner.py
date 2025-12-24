
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data (list or np.array): Input data
        column (int or str): Column index or name if data is structured
        
    Returns:
        np.array: Data with outliers removed
    """
    if isinstance(data, list):
        data = np.array(data)
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return filtered_data

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data (np.array): Input data
        
    Returns:
        dict: Dictionary containing mean, median, and standard deviation
    """
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    
    return stats

def clean_dataset(data, columns=None):
    """
    Clean entire dataset by removing outliers from specified columns.
    
    Args:
        data (np.array): 2D array of data
        columns (list): List of column indices to clean
        
    Returns:
        np.array: Cleaned dataset
    """
    if columns is None:
        columns = range(data.shape[1])
    
    cleaned_data = data.copy()
    
    for col in columns:
        col_data = data[:, col]
        cleaned_col = remove_outliers_iqr(col_data, col)
        
        # For simplicity, pad with mean if lengths don't match
        if len(cleaned_col) < len(col_data):
            mean_val = np.mean(cleaned_col)
            padding = np.full(len(col_data) - len(cleaned_col), mean_val)
            cleaned_col = np.concatenate([cleaned_col, padding])
        
        cleaned_data[:, col] = cleaned_col
    
    return cleaned_data

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.randn(100, 3) * 10 + 50
    print("Original data shape:", sample_data.shape)
    
    cleaned = clean_dataset(sample_data)
    print("Cleaned data shape:", cleaned.shape)
    
    stats = calculate_statistics(cleaned[:, 0])
    print("Column 1 statistics:", stats)import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and save the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Drop rows where all values are NaN
        df = df.dropna(how='all')
        print(f"After dropping all-NaN rows: {df.shape}")
        
        # Fill numeric column NaN values with column mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
        
        # Fill string column NaN values with 'Unknown'
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna('Unknown')
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        # Save cleaned data
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        else:
            base_name = file_path.rsplit('.', 1)[0]
            output_path = f"{base_name}_cleaned.csv"
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate the structure and content of a DataFrame.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            print(f"Column {col} contains infinite values")
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['Alice', 'Bob', 'Alice', None, 'Charlie'],
        'age': [25, 30, 25, 40, None],
        'score': [85.5, 92.0, 85.5, 78.5, 88.0],
        'city': ['NYC', 'LA', 'NYC', 'Chicago', None]
    }
    
    # Create a sample DataFrame for testing
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    # Clean the sample data
    cleaned_df = clean_csv_data('test_data.csv', 'test_data_cleaned.csv')
    
    if cleaned_df is not None:
        print("\nCleaned DataFrame:")
        print(cleaned_df)
        
        # Validate the cleaned data
        is_valid = validate_dataframe(cleaned_df, ['name', 'age', 'score'])
        print(f"\nData validation: {'Passed' if is_valid else 'Failed'}")
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
    
    return filtered_df

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 10 + 50
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[95:99, 'value'] = [200, -100, 300, 150, 250]
    
    print("Original data shape:", df.shape)
    print("\nOriginal statistics:")
    print(calculate_summary_stats(df, 'value'))
    
    cleaned_df = clean_numeric_data(df, ['value'])
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    print(calculate_summary_stats(cleaned_df, 'value'))