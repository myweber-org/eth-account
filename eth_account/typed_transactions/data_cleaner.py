
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    return cleaned_df.reset_index(drop=True)

def main():
    data = {
        'id': range(1, 21),
        'value': [10, 12, 13, 15, 120, 14, 16, 18, 19, 20,
                  22, 24, 26, 28, 30, 32, 34, 200, 36, 38]
    }
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    
    cleaned_df = clean_dataset(df, ['value'])
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    print(f"\nRemoved {len(df) - len(cleaned_df)} outliers")

if __name__ == "__main__":
    main()
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Columns to consider for duplicates
    keep (str, optional): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
        
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    removed_count = len(df) - len(cleaned_df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list, optional): List of required column names
    
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

def get_data_summary(df):
    """
    Generate a summary of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict()
    }
    
    return summary

def clean_dataset(df, config):
    """
    Main function to clean dataset based on configuration.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    config (dict): Cleaning configuration
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if not validate_dataframe(df, config.get('required_columns')):
        raise ValueError("DataFrame validation failed")
    
    if config.get('remove_duplicates', False):
        df = remove_duplicates(
            df, 
            subset=config.get('duplicate_columns'),
            keep=config.get('keep_duplicates', 'first')
        )
    
    if config.get('clean_numeric', False):
        numeric_cols = config.get('numeric_columns', [])
        df = clean_numeric_columns(df, numeric_cols)
    
    return df
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or np.array): Input data array
    column (int): Column index for 2D data, ignored for 1D data
    
    Returns:
    np.array: Data with outliers removed
    """
    if isinstance(data, list):
        data = np.array(data)
    
    # Handle 2D data
    if data.ndim == 2:
        col_data = data[:, column]
    else:
        col_data = data
    
    # Calculate IQR
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    # Define outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter data
    if data.ndim == 2:
        mask = (col_data >= lower_bound) & (col_data <= upper_bound)
        return data[mask]
    else:
        return col_data[(col_data >= lower_bound) & (col_data <= upper_bound)]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (np.array): Input data array
    
    Returns:
    dict: Dictionary containing mean, median, std, min, max
    """
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    return stats

# Example usage
if __name__ == "__main__":
    # Generate sample data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 100)
    outliers = np.array([200, 250, 300, 350])
    sample_data = np.concatenate([normal_data, outliers])
    
    print("Original data statistics:")
    print(calculate_statistics(sample_data))
    print(f"Original data shape: {sample_data.shape}")
    
    # Remove outliers
    cleaned_data = remove_outliers_iqr(sample_data, 0)
    
    print("\nCleaned data statistics:")
    print(calculate_statistics(cleaned_data))
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Removed {len(sample_data) - len(cleaned_data)} outliers")