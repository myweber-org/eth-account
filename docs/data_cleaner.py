
import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', output_path=None):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Args:
        filepath (str): Path to input CSV file
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame only.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            print("Missing values detected:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count} missing")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            
            if fill_strategy == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_strategy == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_strategy == 'zero':
                df[numeric_cols] = df[numeric_cols].fillna(0)
            elif fill_strategy == 'mode':
                for col in numeric_cols:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
            
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            
            print("Missing values filled using strategy:", fill_strategy)
        else:
            print("No missing values found.")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers in a column using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        pd.Series: Boolean mask of outliers
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame")
        return pd.Series([False] * len(df))
    
    if not np.issubdtype(df[column].dtype, np.number):
        print(f"Column '{column}' is not numeric")
        return pd.Series([False] * len(df))
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    if outliers.any():
        print(f"Found {outliers.sum()} outliers in column '{column}'")
    
    return outliers

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': ['a', 'b', np.nan, 'd', 'e'],
        'D': [10, 20, 30, 40, 50]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='mean', output_path='cleaned_data.csv')
    
    if cleaned_df is not None:
        outliers = detect_outliers_iqr(cleaned_df, 'D')
        print(f"Outlier detection complete. Found {outliers.sum()} outliers in column 'D'")