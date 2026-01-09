import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
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
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'empty_rows': 0,
        'total_rows': len(df)
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_result['missing_columns'] = missing
            validation_result['is_valid'] = False
    
    validation_result['empty_rows'] = df.isnull().all(axis=1).sum()
    
    return validation_result

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_data(cleaned, required_columns=['A', 'B'])
    print("\nValidation Result:")
    print(validation)import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Clean CSV data by handling missing values, duplicates, and standardizing formats.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Remove duplicate rows
        df.drop_duplicates(inplace=True)
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        # Standardize text columns: trim whitespace and convert to lowercase
        for col in categorical_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()
        
        # Remove rows where all values are null
        df.dropna(how='all', inplace=True)
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        
        # Save cleaned data
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        else:
            base_name = file_path.rsplit('.', 1)[0]
            cleaned_path = f"{base_name}_cleaned.csv"
            df.to_csv(cleaned_path, index=False)
            print(f"Cleaned data saved to: {cleaned_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    cleaned_df = clean_csv_data(input_file)
    
    if cleaned_df is not None:
        print(f"Data cleaning completed successfully.")
        print(f"Original shape: Not available (file loaded successfully)")
        print(f"Cleaned shape: {cleaned_df.shape}")
        print(f"Columns: {list(cleaned_df.columns)}")