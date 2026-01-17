
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(file_path, output_path):
    try:
        df = pd.read_csv(file_path)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            df = remove_outliers_iqr(df, col)
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error during cleaning: {e}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing rows with null values
    and dropping duplicate rows.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list, optional): List of columns to check for nulls and duplicates.
                                      If None, checks all columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if columns_to_check is None:
        columns_to_check = df.columns.tolist()
    
    # Remove rows with null values in specified columns
    cleaned_df = df.dropna(subset=columns_to_check)
    
    # Remove duplicate rows based on specified columns
    cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list, optional): List of columns that must be present
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False
    
    return True

def get_cleaning_stats(original_df, cleaned_df):
    """
    Calculate statistics about the cleaning process.
    
    Parameters:
    original_df (pd.DataFrame): Original DataFrame before cleaning
    cleaned_df (pd.DataFrame): Cleaned DataFrame after processing
    
    Returns:
    dict: Dictionary containing cleaning statistics
    """
    stats = {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'rows_removed': len(original_df) - len(cleaned_df),
        'removal_percentage': round((1 - len(cleaned_df)/len(original_df)) * 100, 2) if len(original_df) > 0 else 0
    }
    
    return stats

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 3, 4, 5, 5, 6, None],
#         'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank', 'Grace'],
#         'age': [25, 30, 35, 40, 45, 45, 50, 55]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataset(df, columns_to_check=['id', 'name'])
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     stats = get_cleaning_stats(df, cleaned)
#     print("\nCleaning Statistics:")
#     for key, value in stats.items():
#         print(f"{key}: {value}")
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None):
    """
    Normalize data using Min-Max scaling.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def clean_dataset(df, outlier_columns=None, normalize_columns=None):
    """
    Main function to clean dataset by removing outliers and normalizing.
    """
    df_cleaned = remove_outliers_iqr(df, outlier_columns)
    df_final = normalize_minmax(df_cleaned, normalize_columns)
    return df_final

if __name__ == "__main__":
    sample_data = {
        'feature_a': [1, 2, 3, 4, 100],
        'feature_b': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'B', 'A']
    }
    df_sample = pd.DataFrame(sample_data)
    
    print("Original dataset:")
    print(df_sample)
    
    cleaned_df = clean_dataset(
        df_sample, 
        outlier_columns=['feature_a', 'feature_b'],
        normalize_columns=['feature_a', 'feature_b']
    )
    
    print("\nCleaned dataset:")
    print(cleaned_df)