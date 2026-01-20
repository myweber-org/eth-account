import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_na_threshold=0.5):
    """
    Clean a pandas DataFrame by handling missing values,
    removing duplicates, and standardizing column names.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()
    
    # Standardize column names if mapping is provided
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    # Convert column names to lowercase and replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    
    # Calculate missing value percentage for each column
    missing_percentage = (df_clean.isnull().sum() / len(df_clean)) * 100
    
    # Drop columns with too many missing values
    columns_to_drop = missing_percentage[missing_percentage > drop_na_threshold * 100].index
    df_clean = df_clean.drop(columns=columns_to_drop)
    
    # Fill missing values for numeric columns with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Fill missing values for categorical columns with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'unknown')
    
    # Remove outliers using IQR method for numeric columns
    for col in numeric_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
    
    # Generate cleaning report
    report = {
        'original_rows': len(df),
        'cleaned_rows': len(df_clean),
        'duplicates_removed': duplicates_removed,
        'columns_dropped': list(columns_to_drop),
        'missing_values_filled': df.isnull().sum().sum() - df_clean.isnull().sum().sum(),
        'remaining_missing_values': df_clean.isnull().sum().sum()
    }
    
    return df_clean, report

def validate_dataframe(df, required_columns=None, date_columns=None):
    """
    Validate the structure and content of a DataFrame.
    """
    validation_results = {}
    
    # Check if DataFrame is empty
    validation_results['is_empty'] = df.empty
    
    # Check for required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_columns
        validation_results['has_all_required_columns'] = len(missing_columns) == 0
    
    # Validate date columns if specified
    if date_columns:
        invalid_dates = {}
        for col in date_columns:
            if col in df.columns:
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    invalid_count = df[col].isna().sum()
                    invalid_dates[col] = invalid_count
                except:
                    invalid_dates[col] = len(df)
        validation_results['invalid_dates'] = invalid_dates
    
    # Check data types
    validation_results['dtypes'] = df.dtypes.to_dict()
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinite_counts = {}
    for col in numeric_cols:
        if col in df.columns:
            infinite_count = np.isinf(df[col]).sum()
            if infinite_count > 0:
                infinite_counts[col] = infinite_count
    validation_results['infinite_values'] = infinite_counts
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'Customer ID': [1, 2, 3, 4, 5, 5, 6],
        'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'Age': [25, 30, None, 40, 35, 35, 1000],  # 1000 is an outlier
        'Salary': [50000, 60000, 70000, 80000, None, 90000, 95000],
        'Join Date': ['2020-01-01', '2021-02-15', 'invalid', '2022-03-20', '2023-04-10', '2023-04-10', '2024-05-05']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df, report = clean_dataset(df, drop_na_threshold=0.3)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaning Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned data
    validation = validate_dataframe(
        cleaned_df,
        required_columns=['customer_id', 'name', 'age', 'salary'],
        date_columns=['join_date']
    )
    print("Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")