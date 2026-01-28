import pandas as pd
import numpy as np

def clean_csv_data(filepath, output_path=None):
    """
    Load and clean CSV data by handling missing values and converting data types.
    
    Args:
        filepath (str): Path to input CSV file
        output_path (str, optional): Path to save cleaned data. If None, returns DataFrame
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None
    """
    try:
        df = pd.read_csv(filepath)
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        # Convert date columns if present
        date_columns = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_columns.append(col)
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        # Remove rows where all values are NaN
        df.dropna(how='all', inplace=True)
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        
        # Summary statistics
        print(f"Data cleaning completed:")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Processed {len(numeric_cols)} numeric columns")
        print(f"  - Processed {len(categorical_cols)} categorical columns")
        if date_columns:
            print(f"  - Converted {len(date_columns)} date columns")
        print(f"  - Final dataset shape: {df.shape}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Validation failed: Missing required columns: {missing_columns}")
            return False
    
    # Check for any remaining NaN values
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"Warning: DataFrame contains {nan_count} NaN values")
    
    return True

# Example usage
if __name__ == "__main__":
    # Test with sample data creation
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'age': [25, 30, None, 35, 40, 40],
        'salary': [50000, 60000, 70000, None, 90000, 90000],
        'join_date': ['2020-01-15', '2019-03-20', None, '2021-07-10', '2018-11-05', '2018-11-05']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data('test_data.csv')
    
    if cleaned_df is not None:
        # Validate the cleaned data
        is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age'])
        print(f"Data validation result: {'Passed' if is_valid else 'Failed'}")