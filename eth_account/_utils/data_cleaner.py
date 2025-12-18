import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values
                               Options: 'mean', 'median', 'drop', 'zero'
    """
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        df_cleaned = df.copy()
        
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"After removing duplicates: {df_cleaned.shape}")
        
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
        elif missing_strategy == 'median':
            for col in numeric_cols:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        elif missing_strategy == 'zero':
            df_cleaned.fillna(0, inplace=True)
        elif missing_strategy == 'drop':
            df_cleaned.dropna(inplace=True)
        
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_cleaned[col].isnull().any():
                df_cleaned[col].fillna('Unknown', inplace=True)
        
        print(f"Final cleaned data shape: {df_cleaned.shape}")
        print(f"Missing values after cleaning:\n{df_cleaned.isnull().sum().sum()}")
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df, required_columns=None):
    """
    Validate cleaned data for basic quality checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': []
    }
    
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty or None')
        return validation_results
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing required columns: {missing_cols}')
    
    if df.isnull().sum().sum() > 0:
        validation_results['is_valid'] = False
        validation_results['issues'].append('Data contains missing values')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].min() < 0 and col not in ['temperature', 'balance']:
            validation_results['issues'].append(f'Column {col} contains negative values')
    
    return validation_results

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_data = clean_csv_data(input_csv, output_csv, missing_strategy='mean')
    
    if cleaned_data is not None:
        validation = validate_data(cleaned_data)
        print(f"Data validation passed: {validation['is_valid']}")
        if not validation['is_valid']:
            print(f"Validation issues: {validation['issues']}")