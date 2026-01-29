
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    file_path (str): Path to input CSV file.
    output_path (str): Path for cleaned CSV output. If None, returns DataFrame.
    missing_strategy (str): Strategy for handling missing values: 
                           'mean', 'median', 'mode', or 'drop'.
    
    Returns:
    pd.DataFrame or None: Cleaned DataFrame if output_path is None.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        original_rows = df.shape[0]
        df = df.drop_duplicates()
        removed_duplicates = original_rows - df.shape[0]
        print(f"Removed {removed_duplicates} duplicate rows")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'drop':
            df = df.dropna()
            print("Removed rows with missing values")
        elif missing_strategy in ['mean', 'median', 'mode']:
            for col in numeric_cols:
                if df[col].isnull().any():
                    if missing_strategy == 'mean':
                        fill_value = df[col].mean()
                    elif missing_strategy == 'median':
                        fill_value = df[col].median()
                    else:
                        fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
                    
                    df[col].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in '{col}' with {missing_strategy}: {fill_value:.2f}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            print(f"Final data shape: {df.shape}")
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    print(f"DataFrame validation passed. Shape: {df.shape}, Columns: {list(df.columns)}")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'value': [10.5, 20.3, np.nan, 15.7, 20.3, np.nan, 25.1, 30.4],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', missing_strategy='mean')
    
    if cleaned_df is not None:
        validation_result = validate_dataframe(cleaned_df, ['id', 'value', 'category'])
        print(f"Validation result: {validation_result}")
        
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None):
    """
    Remove outliers from specified columns using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, process all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize
    method (str): Normalization method - 'minmax' or 'zscore'
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_norm = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df_norm[col] = (df[col] - mean_val) / std_val
    
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values - 'mean', 'median', 'mode', or 'drop'
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean':
            df_processed[col].fillna(df[col].mean(), inplace=True)
        elif strategy == 'median':
            df_processed[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'mode':
            df_processed[col].fillna(df[col].mode()[0], inplace=True)
        elif strategy == 'drop':
            df_processed = df_processed.dropna(subset=[col])
    
    return df_processed.reset_index(drop=True)