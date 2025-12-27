
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                else:  # mode
                    fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
                
                cleaned_df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col].fillna('Unknown', inplace=True)
            print(f"Filled missing values in categorical column '{col}' with 'Unknown'")
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    print("DataFrame validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', np.nan, 'A'],
        'score': [85, 92, 92, 78, np.nan, 88]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    if validate_dataframe(df, required_columns=['id', 'value']):
        cleaned_df = clean_dataframe(df, drop_duplicates=True, fill_missing='mean')
        print("\nCleaned DataFrame:")
        print(cleaned_df)
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_strategy='mean', output_path=None):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Parameters:
    file_path (str): Path to input CSV file.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'drop').
    output_path (str): Optional path to save cleaned CSV.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_rows = df.shape[0]
    original_cols = df.shape[1]
    
    if fill_strategy == 'drop':
        df_cleaned = df.dropna()
        rows_dropped = original_rows - df_cleaned.shape[0]
        print(f"Dropped {rows_dropped} rows with missing values.")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        
        df_cleaned = df.copy()
        
        for col in numeric_cols:
            if df[col].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = df[col].mean()
                elif fill_strategy == 'median':
                    fill_value = df[col].median()
                elif fill_strategy == 'mode':
                    fill_value = df[col].mode()[0]
                else:
                    raise ValueError(f"Unsupported fill strategy: {fill_strategy}")
                
                df_cleaned[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in column '{col}' with {fill_strategy}: {fill_value}")
        
        for col in non_numeric_cols:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df_cleaned[col].fillna(mode_value, inplace=True)
                print(f"Filled missing values in column '{col}' with mode: {mode_value}")
    
    if output_path:
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    print(f"Original data: {original_rows} rows, {original_cols} columns")
    print(f"Cleaned data: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'empty_rows': 0,
        'duplicate_rows': 0,
        'issues': []
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing}")
    
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows > 0:
        validation_results['empty_rows'] = empty_rows
        validation_results['issues'].append(f"Found {empty_rows} completely empty rows")
    
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        validation_results['duplicate_rows'] = duplicate_rows
        validation_results['issues'].append(f"Found {duplicate_rows} duplicate rows")
    
    return validation_results

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, np.nan, 8, 10],
        'C': ['x', 'y', 'z', np.nan, 'x'],
        'D': [100, 200, 300, 400, 500]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='mean', output_path='cleaned_data.csv')
    
    validation = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C', 'D'])
    print("\nData Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")