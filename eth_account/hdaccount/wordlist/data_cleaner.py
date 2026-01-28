
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean the dataset by removing duplicates and handling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Handle missing values
    # For numerical columns, fill with median
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    # For categorical columns, fill with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown')
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_dataset(df):
    """
    Validate the dataset for common issues.
    """
    validation_report = {}
    
    # Check for remaining missing values
    missing_values = df.isnull().sum().sum()
    validation_report['missing_values'] = missing_values
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    validation_report['duplicates'] = duplicates
    
    # Check data types
    validation_report['dtypes'] = df.dtypes.to_dict()
    
    # Check shape
    validation_report['shape'] = df.shape
    
    return validation_report

def main():
    # Example usage
    data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, None, 35, 40, 40, 45],
        'score': [85.5, 90.0, 78.5, None, 92.5, 92.5, 88.0]
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    print("\nDataset info:")
    print(df.info())
    
    # Clean the dataset
    cleaned_df = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    # Validate the cleaned dataset
    validation = validate_dataset(cleaned_df)
    print("\nValidation report:")
    for key, value in validation.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()