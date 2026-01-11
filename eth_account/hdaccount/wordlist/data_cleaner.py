
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        columns_to_check (list, optional): Specific columns to check for duplicates. 
                                          If None, checks all columns.
        fill_missing (str): Method to fill missing values - 'mean', 'median', or 'drop'
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Remove duplicate rows
    if columns_to_check:
        df_cleaned = df.drop_duplicates(subset=columns_to_check)
    else:
        df_cleaned = df.drop_duplicates()
    
    # Handle missing values
    if fill_missing == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if fill_missing == 'mean':
                fill_value = df_cleaned[col].mean()
            else:  # median
                fill_value = df_cleaned[col].median()
            
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    
    # Report cleaning statistics
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Missing values handled using: {fill_missing}")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
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
            print(f"Validation failed: Missing columns {missing_cols}")
            return False
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
#         'category': ['A', 'B', 'B', 'C', 'A', 'B']
#     }
#     
#     df = pd.DataFrame(data)
#     cleaned_df = clean_dataset(df, columns_to_check=['id'], fill_missing='mean')
#     
#     if validate_dataframe(cleaned_df, required_columns=['id', 'value']):
#         print("Data validation passed")
#         print(cleaned_df)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name.
    
    Returns:
    dict: Dictionary containing statistics.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_clean = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[10, 'A'] = 1000
    df.loc[20, 'B'] = 500
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics for column 'A':")
    print(calculate_statistics(df, 'A'))
    
    cleaned_df = clean_dataset(df, ['A', 'B'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned statistics for column 'A':")
    print(calculate_statistics(cleaned_df, 'A'))