
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
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

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each cleaned column
    """
    cleaned_df = df.copy()
    statistics = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            statistics[column] = stats
    
    return cleaned_df, statistics
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, handle_nulls='drop', null_threshold=0.5):
    """
    Clean a pandas DataFrame by removing duplicates and handling null values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to remove duplicate rows
    handle_nulls (str): Strategy for handling nulls - 'drop', 'fill', or 'threshold'
    null_threshold (float): Threshold for column null percentage when handle_nulls='threshold'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if handle_nulls == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with any null values")
    elif handle_nulls == 'fill':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        object_cols = cleaned_df.select_dtypes(include=['object']).columns
        cleaned_df[object_cols] = cleaned_df[object_cols].fillna('Unknown')
        print("Filled null values with appropriate defaults")
    elif handle_nulls == 'threshold':
        null_percentages = cleaned_df.isnull().sum() / len(cleaned_df)
        cols_to_drop = null_percentages[null_percentages > null_threshold].index
        cleaned_df = cleaned_df.drop(columns=cols_to_drop)
        print(f"Dropped {len(cols_to_drop)} columns with >{null_threshold*100}% null values")
    
    print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, numeric_ranges=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    numeric_ranges (dict): Dictionary mapping column names to (min, max) tuples
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'out_of_range': [],
        'null_counts': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    if numeric_ranges:
        for col, (min_val, max_val) in numeric_ranges.items():
            if col in df.columns:
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(out_of_range) > 0:
                    validation_results['out_of_range'].append({
                        'column': col,
                        'count': len(out_of_range),
                        'min_allowed': min_val,
                        'max_allowed': max_val
                    })
                    validation_results['is_valid'] = False
    
    null_counts = df.isnull().sum()
    validation_results['null_counts'] = null_counts[null_counts > 0].to_dict()
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A'],
        'score': [100, 200, 200, 300, 400, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, handle_nulls='fill')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_data(
        cleaned,
        required_columns=['id', 'value', 'category'],
        numeric_ranges={'score': (0, 1000)}
    )
    
    print("\nValidation Results:")
    print(validation)