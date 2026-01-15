import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', output_path=None):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Args:
        filepath (str): Path to the input CSV file.
        fill_method (str): Method to fill missing values ('mean', 'median', 'mode', or 'zero').
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values.")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if fill_method == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_method == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_method == 'mode':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
            elif fill_method == 'zero':
                df[numeric_cols] = df[numeric_cols].fillna(0)
            else:
                raise ValueError("Invalid fill_method. Choose from 'mean', 'median', 'mode', or 'zero'.")
            
            print(f"Missing values filled using '{fill_method}' method.")
        else:
            print("No missing values found.")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None

def validate_dataframe(df, check_duplicates=True, check_negative=True):
    """
    Perform basic validation on a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        check_duplicates (bool): Check for duplicate rows.
        check_negative (bool): Check for negative values in numeric columns.
    
    Returns:
        dict: Dictionary containing validation results.
    """
    validation_results = {}
    
    if df is None or df.empty:
        validation_results['error'] = 'DataFrame is empty or None'
        return validation_results
    
    validation_results['row_count'] = len(df)
    validation_results['column_count'] = len(df.columns)
    
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        validation_results['duplicate_rows'] = duplicate_count
    
    if check_negative:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        negative_counts = {}
        for col in numeric_cols:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                negative_counts[col] = negative_count
        validation_results['negative_values'] = negative_counts
    
    return validation_results

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, np.nan, 8, 10],
        'C': ['x', 'y', 'z', 'x', 'y']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_method='mean')
    
    if cleaned_df is not None:
        validation = validate_dataframe(cleaned_df)
        print("Validation results:", validation)
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop').
    outlier_threshold (float): Number of standard deviations to consider as outlier.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    for column in cleaned_df.select_dtypes(include=[np.number]).columns:
        if cleaned_df[column].isnull().any():
            if missing_strategy == 'mean':
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif missing_strategy == 'median':
                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif missing_strategy == 'mode':
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            elif missing_strategy == 'drop':
                cleaned_df.dropna(subset=[column], inplace=True)
    
    # Remove outliers using z-score method
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((cleaned_df[numeric_columns] - cleaned_df[numeric_columns].mean()) / 
                     cleaned_df[numeric_columns].std())
    
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    cleaned_df = cleaned_df[outlier_mask].reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (bool, str) indicating validation result and message.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 100],  # Contains NaN and outlier
        'B': [5, 6, 7, np.nan, 8],
        'C': [9, 10, 11, 12, 13]
    })
    
    print("Original data:")
    print(sample_data)
    
    # Clean the data
    cleaned_data = clean_dataset(sample_data, missing_strategy='mean', outlier_threshold=2)
    
    print("\nCleaned data:")
    print(cleaned_data)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned_data, required_columns=['A', 'B', 'C'], min_rows=2)
    print(f"\nValidation: {is_valid}, Message: {message}")