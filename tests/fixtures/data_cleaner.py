import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and optionally dropping columns.
    
    Args:
        filepath: Path to the CSV file
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        columns_to_drop: List of column names to drop (optional)
    
    Returns:
        Cleaned pandas DataFrame
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {filepath}")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"Dropped columns: {columns_to_drop}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(f"Column '{col}' has {missing_count} missing values")
            
            if col in numeric_cols:
                if missing_strategy == 'mean':
                    fill_value = df[col].mean()
                elif missing_strategy == 'median':
                    fill_value = df[col].median()
                elif missing_strategy == 'mode':
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
                else:
                    df = df.dropna(subset=[col])
                    continue
                
                df[col] = df[col].fillna(fill_value)
                print(f"  Filled with {missing_strategy}: {fill_value:.2f}")
            
            elif col in categorical_cols:
                if missing_strategy == 'mode':
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(fill_value)
                    print(f"  Filled with mode: '{fill_value}'")
                else:
                    df = df.dropna(subset=[col])
    
    final_shape = df.shape
    rows_removed = original_shape[0] - final_shape[0]
    cols_removed = original_shape[1] - final_shape[1]
    
    print(f"Final data shape: {final_shape}")
    print(f"Rows removed: {rows_removed}, Columns removed: {cols_removed}")
    
    return df

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    
    Args:
        df: DataFrame containing the data
        column: Column name to check for outliers
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        Boolean mask indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    outlier_count = outliers.sum()
    if outlier_count > 0:
        print(f"Detected {outlier_count} outliers in column '{column}'")
        print(f"Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    return outliers

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    
    Args:
        df: Cleaned DataFrame
        output_path: Path to save the cleaned data
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['X', 'Y', np.nan, 'X', 'Y', 'Z'],
        'D': [100, 200, 300, 400, 500, 600]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', missing_strategy='mean', columns_to_drop=['D'])
    
    outliers = detect_outliers_iqr(cleaned_df, 'A')
    print(f"Outlier indices: {cleaned_df[outliers].index.tolist()}")
    
    save_cleaned_data(cleaned_df, 'cleaned_sample_data.csv')