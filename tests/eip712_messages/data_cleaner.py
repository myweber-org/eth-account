
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    if fill_missing:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if fill_missing == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_missing == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_missing == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values remain in non-numeric columns")
    
    print(f"Dataset cleaned: {original_shape} -> {df.shape}")
    return df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nCleaning dataset...")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    try:
        validate_dataset(cleaned_df, required_columns=['id', 'value'])
        print("\nDataset validation passed")
    except ValueError as e:
        print(f"\nDataset validation failed: {e}")
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, fill_strategy='mean'):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Args:
        file_path (str): Path to input CSV file.
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'drop').
    
    Returns:
        pandas.DataFrame or None: Cleaned DataFrame if output_path is None, else None.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            if fill_strategy == 'drop':
                df_cleaned = df.dropna()
                print(f"Dropped rows with missing values. New shape: {df_cleaned.shape}")
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if df[col].isnull().any():
                        if fill_strategy == 'mean':
                            fill_value = df[col].mean()
                        elif fill_strategy == 'median':
                            fill_value = df[col].median()
                        elif fill_strategy == 'mode':
                            fill_value = df[col].mode()[0]
                        else:
                            fill_value = 0
                        
                        df[col] = df[col].fillna(fill_value)
                        print(f"Filled missing values in '{col}' with {fill_strategy}: {fill_value}")
                
                # For non-numeric columns, fill with most frequent value
                non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
                for col in non_numeric_cols:
                    if df[col].isnull().any():
                        mode_value = df[col].mode()[0] if not df[col].mode().empty else ''
                        df[col] = df[col].fillna(mode_value)
                        print(f"Filled missing values in '{col}' with mode: '{mode_value}'")
                
                df_cleaned = df
        else:
            df_cleaned = df
            print("No missing values found")
        
        # Remove duplicate rows
        duplicates = df_cleaned.duplicated().sum()
        if duplicates > 0:
            df_cleaned = df_cleaned.drop_duplicates()
            print(f"Removed {duplicates} duplicate rows")
        
        # Reset index after cleaning
        df_cleaned = df_cleaned.reset_index(drop=True)
        
        if output_path:
            df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df_cleaned
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pandas.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        inf_count = np.isinf(numeric_df).sum().sum()
        if inf_count > 0:
            print(f"Warning: Found {inf_count} infinite values in numeric columns")
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, np.nan, 15.2, 15.2, np.nan],
        'category': ['A', 'B', 'A', 'A', 'B'],
        'score': [100, 200, np.nan, 300, 400]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='mean')
    
    if cleaned_df is not None:
        validation_passed = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
        
        if validation_passed:
            print("Data cleaning and validation completed successfully")
            print("\nCleaned DataFrame:")
            print(cleaned_df)
    
    # Clean up sample file
    import os
    if os.path.exists('sample_data.csv'):
        os.remove('sample_data.csv')