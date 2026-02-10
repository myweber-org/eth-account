import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if cleaned_df.isnull().sum().any():
        print("Handling missing values...")
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        if fill_missing == 'mean':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        elif fill_missing == 'median':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        elif fill_missing == 'zero':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)
        else:
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values.")
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame structure and content.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_strategy='mean'):
    """
    Load a CSV file, handle missing values, and return cleaned DataFrame.
    
    Parameters:
    file_path (str): Path to the CSV file.
    fill_strategy (str): Strategy for filling missing values. 
                         Options: 'mean', 'median', 'mode', 'zero'.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("The provided CSV file is empty.")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        print("Missing values per column:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count} missing")
        
        if fill_strategy == 'mean':
            df = df.fillna(df.select_dtypes(include=[np.number]).mean())
        elif fill_strategy == 'median':
            df = df.fillna(df.select_dtypes(include=[np.number]).median())
        elif fill_strategy == 'mode':
            df = df.fillna(df.mode().iloc[0])
        elif fill_strategy == 'zero':
            df = df.fillna(0)
        else:
            raise ValueError(f"Unsupported fill strategy: {fill_strategy}")
        
        print(f"Missing values filled using '{fill_strategy}' strategy.")
    else:
        print("No missing values found.")
    
    cleaned_shape = df.shape
    print(f"Cleaned data shape: {cleaned_shape}")
    
    if original_shape != cleaned_shape:
        print("Warning: Data shape changed during cleaning.")
    
    return df

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specific column using IQR method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    column (str): Column name to process.
    
    Returns:
    pandas.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    if not np.issubdtype(df[column].dtype, np.number):
        raise ValueError(f"Column '{column}' must be numeric.")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    original_len = len(df)
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = original_len - len(df_clean)
    
    if removed_count > 0:
        print(f"Removed {removed_count} outliers from column '{column}'.")
    
    return df_clean

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to save.
    output_path (str): Path for output CSV file.
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    except Exception as e:
        raise IOError(f"Failed to save data: {str(e)}")

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': [100, 200, 300, np.nan, 500]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    import os
    if os.path.exists('sample_data.csv'):
        os.remove('sample_data.csv')