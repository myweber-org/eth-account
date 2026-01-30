import pandas as pd
import numpy as np

def clean_csv_data(file_path, strategy='mean', columns=None):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        file_path (str): Path to the CSV file.
        strategy (str): Strategy for filling missing values.
                        Options: 'mean', 'median', 'mode', 'drop'.
        columns (list): List of column names to apply cleaning to.
                        If None, applies to all numeric columns.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df[col].mean()
        elif strategy == 'median':
            fill_value = df[col].median()
        elif strategy == 'mode':
            fill_value = df[col].mode()[0] if not df[col].mode().empty else np.nan
        elif strategy == 'drop':
            df = df.dropna(subset=[col])
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df[col] = df[col].fillna(fill_value)
    
    return df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a column using the IQR method.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        column (str): Column name to process.
        multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
        pandas.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, np.nan, 30, 40, 50, 60],
        'C': ['a', 'b', 'c', 'd', 'e', 'f']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    cleaned = clean_csv_data('sample_data.csv', strategy='median', columns=['A', 'B'])
    print("Cleaned DataFrame:")
    print(cleaned)
    
    filtered = remove_outliers_iqr(cleaned, 'A')
    print("\nDataFrame after outlier removal:")
    print(filtered)