import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                else:  # mode
                    fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
        return validation_results
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_columns = null_counts[null_counts > 0].index.tolist()
        validation_results['warnings'].append(f"Found missing values in columns: {null_columns}")
    
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        validation_results['warnings'].append(f"Found {duplicate_count} duplicate rows")
    
    return validation_results

def normalize_columns(df, columns=None):
    """
    Normalize specified columns to have values between 0 and 1.
    
    Args:
        df: pandas DataFrame
        columns: List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    normalized_df = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max > col_min:  # Avoid division by zero
                normalized_df[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
    
    return normalized_df
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values. 
                            Options: 'mean', 'median', 'mode', 'drop'.
    outlier_threshold (float): Number of standard deviations for outlier detection.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    else:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if missing_strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif missing_strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif missing_strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                else:
                    raise ValueError(f"Unknown missing strategy: {missing_strategy}")
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
    
    # Handle outliers for numerical columns
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        if cleaned_df[column].dtype in [np.float64, np.int64]:
            mean = cleaned_df[column].mean()
            std = cleaned_df[column].std()
            
            # Identify outliers
            outliers = np.abs(cleaned_df[column] - mean) > outlier_threshold * std
            
            # Cap outliers to threshold
            upper_bound = mean + outlier_threshold * std
            lower_bound = mean - outlier_threshold * std
            
            cleaned_df.loc[outliers & (cleaned_df[column] > upper_bound), column] = upper_bound
            cleaned_df.loc[outliers & (cleaned_df[column] < lower_bound), column] = lower_bound
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'A': [1, 2, np.nan, 4, 100],  # Contains NaN and outlier
#         'B': [5, 6, 7, 8, 9],
#         'C': [10, 11, 12, 13, 14]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataframe(df, missing_strategy='mean', outlier_threshold=2)
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
#     print(f"\nValidation: {is_valid}, Message: {message}")import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
        subset (list): Columns to consider for duplicates (optional)
        keep (str): Which duplicates to keep: 'first', 'last', or False
    
    Returns:
        DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
        final_count = len(cleaned_df)
        
        duplicates_removed = initial_count - final_count
        
        if output_file:
            cleaned_df.to_csv(output_file, index=False)
            print(f"Processed {input_file}")
            print(f"Removed {duplicates_removed} duplicate rows")
            print(f"Saved cleaned data to {output_file}")
        else:
            print(f"Removed {duplicates_removed} duplicate rows")
            print(f"Original rows: {initial_count}, Cleaned rows: {final_count}")
        
        return cleaned_df
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = original_shape[0] - cleaned_df.shape[0]
        print(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    missing_before = cleaned_df.isnull().sum().sum()
    
    if missing_before > 0:
        if fill_missing == 'mean':
            # Fill numeric columns with mean, categorical with mode
            for column in cleaned_df.columns:
                if cleaned_df[column].dtype in [np.float64, np.int64]:
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                else:
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown', inplace=True)
        
        elif fill_missing == 'median':
            # Fill numeric columns with median, categorical with mode
            for column in cleaned_df.columns:
                if cleaned_df[column].dtype in [np.float64, np.int64]:
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                else:
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown', inplace=True)
        
        elif fill_missing == 'mode':
            # Fill all columns with mode
            for column in cleaned_df.columns:
                if not cleaned_df[column].mode().empty:
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
                else:
                    cleaned_df[column].fillna('Unknown', inplace=True)
        
        elif fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
        
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Handled {missing_before - missing_after} missing values using '{fill_missing}' method")
    
    print(f"Data cleaning complete. Original shape: {original_shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of columns that must be present
        numeric_columns (list): List of columns that should be numeric
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check for required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check numeric columns
    if numeric_columns:
        for column in numeric_columns:
            if column in df.columns:
                if not pd.api.types.is_numeric_dtype(df[column]):
                    validation_results['warnings'].append(f"Column '{column}' is not numeric")
    
    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        infinite_count = np.isinf(numeric_df).sum().sum()
        if infinite_count > 0:
            validation_results['warnings'].append(f"Found {infinite_count} infinite values in numeric columns")
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data with issues
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', None],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataframe(df, drop_duplicates=True, fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    validation = validate_dataframe(
        cleaned_df,
        required_columns=['id', 'name', 'age', 'score'],
        numeric_columns=['age', 'score']
    )
    
    print("\nValidation Results:")
    print(f"Is valid: {validation['is_valid']}")
    if validation['errors']:
        print("Errors:", validation['errors'])
    if validation['warnings']:
        print("Warnings:", validation['warnings'])