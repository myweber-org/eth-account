
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specific column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count()
    }
    
    return stats

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data

if __name__ == "__main__":
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'])
    print("Cleaned data shape:", cleaned.shape)
    
    for col in ['A', 'B', 'C']:
        stats = calculate_statistics(cleaned, col)
        print(f"\nStatistics for {col}:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")