
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or np.array): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    np.array: Data with outliers removed.
    """
    data = np.array(data)
    col_data = data[:, column].astype(float)
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Parameters:
    data (list or np.array): The dataset.
    columns_to_clean (list): List of column indices to clean.
    
    Returns:
    np.array: Cleaned dataset.
    """
    cleaned_data = np.array(data)
    for col in columns_to_clean:
        cleaned_data = remove_outliers_iqr(cleaned_data, col)
    return cleaned_data

if __name__ == "__main__":
    sample_data = [
        [1, 150.5, 'A'],
        [2, 200.0, 'B'],
        [3, 50.0, 'C'],
        [4, 300.0, 'D'],
        [5, 180.0, 'E'],
        [6, 900.0, 'F'],
        [7, 190.0, 'G']
    ]
    
    print("Original data:")
    for row in sample_data:
        print(row)
    
    cleaned = clean_dataset(sample_data, [1])
    
    print("\nCleaned data:")
    for row in cleaned:
        print(row)