
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: Input pandas DataFrame
        drop_duplicates: Boolean flag to remove duplicate rows
        fill_missing: Boolean flag to fill missing values
        fill_strategy: Strategy for filling missing values ('mean', 'median', 'mode', or 'zero')
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_rows = initial_rows - len(cleaned_df)
        print(f"Removed {removed_rows} duplicate rows")
    
    if fill_missing:
        missing_count = cleaned_df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            for column in cleaned_df.columns:
                if cleaned_df[column].dtype in ['int64', 'float64']:
                    if fill_strategy == 'mean':
                        cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                    elif fill_strategy == 'median':
                        cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                    elif fill_strategy == 'mode':
                        cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
                    elif fill_strategy == 'zero':
                        cleaned_df[column].fillna(0, inplace=True)
                else:
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            
            print(f"Missing values filled using {fill_strategy} strategy")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df: Input pandas DataFrame
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required
    
    Returns:
        Boolean indicating if data is valid
    """
    if len(df) < min_rows:
        print(f"DataFrame has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def generate_summary(df):
    """
    Generate a summary report for the DataFrame.
    
    Args:
        df: Input pandas DataFrame
    
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': len(df) - len(df.drop_duplicates())
    }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, 20.3, np.nan, 40.1, 50.0, 50.0, np.nan],
        'category': ['A', 'B', 'A', 'C', 'B', 'B', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, fill_strategy='mean')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    is_valid = validate_data(cleaned_df, required_columns=['id', 'value', 'category'])
    print(f"Data validation result: {is_valid}")
    
    summary = generate_summary(cleaned_df)
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")