
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    method (str): Outlier detection method ('iqr' or 'zscore').
    threshold (float): Threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        mask = z_scores <= threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, fill_strategy='mean', drop_threshold=0.5):
    """
    Clean CSV data by handling missing values and converting data types.
    
    Args:
        file_path: Path to input CSV file
        output_path: Path for cleaned CSV output (optional)
        fill_strategy: Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
        drop_threshold: Drop columns with missing values above this threshold (0.0 to 1.0)
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        
        original_shape = df.shape
        print(f"Original data shape: {original_shape}")
        
        # Drop columns with too many missing values
        missing_ratio = df.isnull().sum() / len(df)
        columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
        df = df.drop(columns=columns_to_drop)
        
        if len(columns_to_drop) > 0:
            print(f"Dropped columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
        
        # Fill missing values based on strategy
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if fill_strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_strategy == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        elif fill_strategy == 'mode':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        
        # Convert object columns to categorical where appropriate
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Convert to categorical if less than 50% unique values
                df[col] = df[col].astype('category')
        
        # Remove duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            print(f"Removed {duplicates} duplicate rows")
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        print(f"Cleaned data shape: {df.shape}")
        print(f"Removed {original_shape[0] - df.shape[0]} rows and {original_shape[1] - df.shape[1]} columns")
        
        # Save to output file if specified
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
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
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty or None')
        return validation_results
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        validation_results['warnings'].append(f'Found {inf_count} infinite values in numeric columns')
    
    # Check data types
    object_cols = df.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        validation_results['warnings'].append(f'Found {len(object_cols)} object columns that may need encoding')
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': ['a', 'b', 'a', 'b', 'c'],
        'D': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', 'cleaned_sample.csv', fill_strategy='mean')
    
    if cleaned_df is not None:
        validation = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
        print(f"Validation results: {validation}")