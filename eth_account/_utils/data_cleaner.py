import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_na_threshold=0.5):
    """
    Clean a pandas DataFrame by removing duplicates, handling missing values,
    and optionally renaming columns.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    
    # Rename columns if mapping is provided
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    # Calculate missing value percentage for each column
    missing_percent = df_clean.isnull().sum() / len(df_clean)
    
    # Drop columns with missing values above threshold
    columns_to_drop = missing_percent[missing_percent > drop_na_threshold].index
    df_clean = df_clean.drop(columns=columns_to_drop)
    
    # Fill remaining missing values with appropriate defaults
    for column in df_clean.columns:
        if df_clean[column].dtype == 'object':
            # For categorical columns, fill with mode
            if not df_clean[column].mode().empty:
                df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
            else:
                df_clean[column] = df_clean[column].fillna('Unknown')
        else:
            # For numerical columns, fill with median
            df_clean[column] = df_clean[column].fillna(df_clean[column].median())
    
    # Remove outliers using IQR method for numerical columns
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing rows
        df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
    
    # Standardize column names: lowercase with underscores
    df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    # Print cleaning summary
    print(f"Cleaning Summary:")
    print(f"- Original rows: {initial_rows}")
    print(f"- Duplicates removed: {duplicates_removed}")
    print(f"- Columns dropped due to high missing values: {list(columns_to_drop)}")
    print(f"- Final dataset shape: {df_clean.shape}")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic quality requirements.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for completely empty columns
    empty_columns = df.columns[df.isnull().all()]
    if len(empty_columns) > 0:
        print(f"Warning: Found completely empty columns: {list(empty_columns)}")
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'Customer ID': [1, 2, 2, 3, 4, 5, None],
        'Name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank', 'Grace'],
        'Age': [25, 30, 30, 35, 28, None, 40],
        'Salary': [50000, 60000, 60000, 75000, 55000, 48000, 90000],
        'Empty Column': [None, None, None, None, None, None, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    column_mapping = {'Customer ID': 'customer_id', 'Name': 'name', 'Age': 'age', 'Salary': 'salary'}
    cleaned_df = clean_dataset(df, column_mapping=column_mapping, drop_na_threshold=0.6)
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    try:
        validate_dataframe(cleaned_df, required_columns=['customer_id', 'name', 'age', 'salary'])
        print("\nData validation passed!")
    except ValueError as e:
        print(f"\nData validation failed: {e}")