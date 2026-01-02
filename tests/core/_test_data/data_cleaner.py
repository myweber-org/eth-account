
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
            elif cleaned_df[column].dtype == 'object':
                cleaned_df[column] = cleaned_df[column].fillna('Unknown')
        print("Missing values have been filled.")
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate the DataFrame for common data quality issues.
    """
    issues = []
    
    if df.empty:
        issues.append("DataFrame is empty.")
    
    for column in df.columns:
        if df[column].isnull().all():
            issues.append(f"Column '{column}' contains only null values.")
    
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate rows.")
    
    return issues

def main():
    sample_data = {
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
        'age': [25, 30, 25, None, 35],
        'score': [85.5, 92.0, 85.5, 78.5, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    issues = validate_dataframe(df)
    if issues:
        print("Data quality issues found:")
        for issue in issues:
            print(f"- {issue}")
    print("\n")
    
    cleaned_df = clean_dataframe(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)

if __name__ == "__main__":
    main()