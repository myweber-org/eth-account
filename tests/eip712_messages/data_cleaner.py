
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    if fill_missing:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if fill_missing == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_missing == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_missing == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values remain in non-numeric columns")
    
    print(f"Dataset cleaned: {original_shape} -> {df.shape}")
    return df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nCleaning dataset...")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    try:
        validate_dataset(cleaned_df, required_columns=['id', 'value'])
        print("\nDataset validation passed")
    except ValueError as e:
        print(f"\nDataset validation failed: {e}")
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, fill_strategy='mean'):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Args:
        file_path (str): Path to input CSV file.
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'drop').
    
    Returns:
        pandas.DataFrame or None: Cleaned DataFrame if output_path is None, else None.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            if fill_strategy == 'drop':
                df_cleaned = df.dropna()
                print(f"Dropped rows with missing values. New shape: {df_cleaned.shape}")
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if df[col].isnull().any():
                        if fill_strategy == 'mean':
                            fill_value = df[col].mean()
                        elif fill_strategy == 'median':
                            fill_value = df[col].median()
                        elif fill_strategy == 'mode':
                            fill_value = df[col].mode()[0]
                        else:
                            fill_value = 0
                        
                        df[col] = df[col].fillna(fill_value)
                        print(f"Filled missing values in '{col}' with {fill_strategy}: {fill_value}")
                
                # For non-numeric columns, fill with most frequent value
                non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
                for col in non_numeric_cols:
                    if df[col].isnull().any():
                        mode_value = df[col].mode()[0] if not df[col].mode().empty else ''
                        df[col] = df[col].fillna(mode_value)
                        print(f"Filled missing values in '{col}' with mode: '{mode_value}'")
                
                df_cleaned = df
        else:
            df_cleaned = df
            print("No missing values found")
        
        # Remove duplicate rows
        duplicates = df_cleaned.duplicated().sum()
        if duplicates > 0:
            df_cleaned = df_cleaned.drop_duplicates()
            print(f"Removed {duplicates} duplicate rows")
        
        # Reset index after cleaning
        df_cleaned = df_cleaned.reset_index(drop=True)
        
        if output_path:
            df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df_cleaned
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pandas.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        inf_count = np.isinf(numeric_df).sum().sum()
        if inf_count > 0:
            print(f"Warning: Found {inf_count} infinite values in numeric columns")
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, np.nan, 15.2, 15.2, np.nan],
        'category': ['A', 'B', 'A', 'A', 'B'],
        'score': [100, 200, np.nan, 300, 400]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='mean')
    
    if cleaned_df is not None:
        validation_passed = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
        
        if validation_passed:
            print("Data cleaning and validation completed successfully")
            print("\nCleaned DataFrame:")
            print(cleaned_df)
    
    # Clean up sample file
    import os
    if os.path.exists('sample_data.csv'):
        os.remove('sample_data.csv')
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistical measures
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean. 
                                 If None, cleans all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10],
        'humidity': [45, 46, 47, 48, 49, 50, 200, 51, 52, 53],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1500, 1021]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nOriginal shape:", df.shape)
    
    cleaned_df = clean_numeric_data(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned shape:", cleaned_df.shape)
    
    for col in df.columns:
        stats = calculate_basic_stats(df, col)
        print(f"\nStatistics for {col}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def process_dataframe(df, column):
    """
    Complete data processing pipeline: remove outliers and return cleaned data with statistics.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    original_stats = calculate_summary_statistics(df, column)
    cleaned_df = remove_outliers_iqr(df, column)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column)
    
    return cleaned_df, original_stats, cleaned_stats

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 15, 12, 11, 10, 9, 8, 12, 13, 14, 200]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50)
    
    try:
        cleaned_df, original_stats, cleaned_stats = process_dataframe(df, 'values')
        
        print("\nCleaned DataFrame:")
        print(cleaned_df)
        
        print("\nOriginal Statistics:")
        for key, value in original_stats.items():
            print(f"{key}: {value:.2f}")
        
        print("\nCleaned Statistics:")
        for key, value in cleaned_stats.items():
            print(f"{key}: {value:.2f}")
            
        print(f"\nRemoved {len(df) - len(cleaned_df)} outliers")
        
    except ValueError as e:
        print(f"Error: {e}")
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        remove_duplicates (bool): If True, remove duplicate rows.
        fill_method (str): Method to handle missing values: 'drop', 'fill_mean', 'fill_median', 'fill_mode'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'fill_mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_method == 'fill_median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_method == 'fill_mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    else:
        raise ValueError("Invalid fill_method. Choose from 'drop', 'fill_mean', 'fill_median', 'fill_mode'.")
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        dict: Dictionary containing validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['all_required_columns_present'] = len(missing_columns) == 0
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, None, 4, 2],
        'B': [5, None, 7, 8, 5],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, remove_duplicates=True, fill_method='fill_mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nCleaned Validation Results:")
    print(validate_dataset(cleaned))