import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(file_path, output_path=None):
    """
    Clean dataset by removing outliers from numerical columns.
    
    Parameters:
    file_path (str): Path to input CSV file
    output_path (str, optional): Path to save cleaned CSV file
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df = pd.read_csv(file_path)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        original_len = len(df)
        df = remove_outliers_iqr(df, col)
        removed_count = original_len - len(df)
        print(f"Removed {removed_count} outliers from column '{col}'")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_dataset(input_file, output_file)
        print(f"Data cleaning complete. Original shape: unknown, Cleaned shape: {cleaned_df.shape}")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing and cleaned_df.isnull().sum().sum() > 0:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                elif fill_strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unsupported fill strategy: {fill_strategy}")
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"Filled missing values in column '{column}' with {fill_strategy}: {fill_value}")
        
        for column in cleaned_df.select_dtypes(exclude=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                cleaned_df[column] = cleaned_df[column].fillna('Unknown')
                print(f"Filled missing values in column '{column}' with 'Unknown'")
    
    print(f"Cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic quality checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of columns that must be present
    
    Returns:
        dict: Dictionary containing validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['has_all_required_columns'] = len(missing_columns) == 0
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Charlie', 'David', None],
        'age': [25, 30, None, 35, 40, 45],
        'score': [85.5, 92.0, 78.5, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, fill_strategy='median')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    validation = validate_dataset(cleaned_df, required_columns=['id', 'name', 'age'])
    print("\nValidation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")