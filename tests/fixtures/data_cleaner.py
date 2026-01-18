
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
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

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 14, 13, 12, 11, 10, 9, 8, 12, 13]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    stats = calculate_summary_statistics(cleaned_df, 'values')
    print("\nSummary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if subset is None:
        subset = dataframe.columns.tolist()
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if dataframe.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def clean_numeric_columns(dataframe, columns=None):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=['object']).columns
    
    cleaned_df = dataframe.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            try:
                cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
            except Exception as e:
                print(f"Error converting column {column}: {e}")
    
    return cleaned_df

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 3, 2, 4, 1],
        'name': ['Alice', 'Bob', 'Charlie', 'Bob', 'David', 'Alice'],
        'value': ['100', '200', '300', '200', '400', '100']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    if validate_dataframe(df, required_columns=['id', 'name']):
        cleaned_df = remove_duplicates(df, subset=['id', 'name'])
        cleaned_df = clean_numeric_columns(cleaned_df, columns=['value'])
        
        print("\nCleaned DataFrame:")
        print(cleaned_df)
        
        return cleaned_df

if __name__ == "__main__":
    main()import pandas as pd

def clean_dataset(df, column_names):
    """
    Remove duplicate rows and fill missing values with column mean.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing values with column mean for numeric columns
    for col in column_names:
        if df_cleaned[col].dtype in ['int64', 'float64']:
            df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
        else:
            df_cleaned[col].fillna('Unknown', inplace=True)
    
    return df_cleaned

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv('raw_data.csv')
    columns_to_clean = ['age', 'salary', 'department']
    cleaned_data = clean_dataset(data, columns_to_clean)
    save_cleaned_data(cleaned_data, 'cleaned_data.csv')