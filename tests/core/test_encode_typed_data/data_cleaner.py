
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean dataset by removing outliers from multiple numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'score': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[10, 'value'] = 500  # Add an outlier
    df.loc[20, 'value'] = -200  # Add another outlier
    
    print("Original dataset statistics:")
    print(calculate_statistics(df, 'value'))
    
    cleaned_df = clean_dataset(df, ['value', 'score'])
    
    print("\nCleaned dataset statistics:")
    print(calculate_statistics(cleaned_df, 'value'))