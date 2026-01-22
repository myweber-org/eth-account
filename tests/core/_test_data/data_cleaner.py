import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to remove duplicate rows.
        fill_missing (str): Strategy to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df.drop_duplicates(inplace=True)
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df.dropna(inplace=True)
        print("Dropped rows with missing values.")
    else:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                else:
                    raise ValueError("fill_missing must be 'mean', 'median', 'mode', or 'drop'")
                
                cleaned_df[column].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{column}' with {fill_missing}: {fill_value}")
    
    for column in cleaned_df.select_dtypes(include=['object']).columns:
        if cleaned_df[column].isnull().any():
            cleaned_df[column].fillna('Unknown', inplace=True)
            print(f"Filled missing values in '{column}' with 'Unknown'")
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame for common data quality issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        dict: Dictionary containing validation results.
    """
    validation_results = {
        'total_rows': df.shape[0],
        'total_columns': df.shape[1],
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    if validation_results['numeric_columns']:
        numeric_stats = {}
        for col in validation_results['numeric_columns']:
            numeric_stats[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
        validation_results['numeric_statistics'] = numeric_stats
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, None],
        'department': ['HR', 'IT', 'IT', 'Finance', None, 'Marketing']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataframe(df))
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nCleaned Validation Results:")
    print(validate_dataframe(cleaned))import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
        
        df.to_csv('cleaned_data.csv', index=False)
        print(f"Data cleaned successfully. Original shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error during cleaning: {e}")
        return None

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv')