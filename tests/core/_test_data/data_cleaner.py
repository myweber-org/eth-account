
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Load a CSV file, clean missing values, convert data types,
    and save the cleaned data to a new file.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Display initial info
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled missing values in {col} with median: {median_val}")
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
                print(f"Converted {col} to datetime format")
            except:
                print(f"Could not convert {col} to datetime")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        if removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        print(f"Final data shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Perform basic validation on the cleaned dataframe.
    """
    if df is None:
        print("DataFrame is None, cannot validate")
        return False
    
    validation_passed = True
    
    # Check for remaining missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Warning: {missing_values} missing values still present")
        validation_passed = False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            print(f"Warning: Infinite values found in column {col}")
            validation_passed = False
    
    # Check data types
    print("\nData types after cleaning:")
    print(df.dtypes)
    
    return validation_passed

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_csv, output_csv)
    
    if cleaned_df is not None:
        print("\nData validation results:")
        is_valid = validate_dataframe(cleaned_df)
        
        if is_valid:
            print("Data validation passed successfully")
        else:
            print("Data validation completed with warnings")