import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. Can be 'mean', 
                                   'median', 'mode', or a dictionary of column:value pairs.
                                   If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataset(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {message}")
import pandas as pd
import numpy as np

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
    
    return filtered_df

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

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of summary statistics for each cleaned column
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    all_stats = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            
            all_stats[column] = stats
    
    return cleaned_df, all_statsimport pandas as pd
import numpy as np
import logging

def clean_csv_data(input_file, output_file, drop_na=True, fill_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_file (str): Path to input CSV file.
        output_file (str): Path to save cleaned CSV file.
        drop_na (bool): Whether to drop rows with missing values.
        fill_strategy (str): Strategy to fill missing values ('mean', 'median', 'mode').
    
    Returns:
        bool: True if cleaning successful, False otherwise.
    """
    try:
        df = pd.read_csv(input_file)
        logging.info(f"Loaded data from {input_file} with shape {df.shape}")
        
        original_rows = len(df)
        df = df.drop_duplicates()
        logging.info(f"Removed {original_rows - len(df)} duplicate rows")
        
        if drop_na:
            df = df.dropna()
            logging.info("Dropped rows with missing values")
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if fill_strategy == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_strategy == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_strategy == 'mode':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
            logging.info(f"Filled missing values using {fill_strategy} strategy")
        
        df.to_csv(output_file, index=False)
        logging.info(f"Saved cleaned data to {output_file}")
        return True
        
    except FileNotFoundError:
        logging.error(f"Input file {input_file} not found")
        return False
    except Exception as e:
        logging.error(f"Error during data cleaning: {str(e)}")
        return False

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pd.DataFrame): Dataframe to validate.
        required_columns (list): List of required column names.
    
    Returns:
        dict: Validation results.
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("Dataframe is empty")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            validation_results['warnings'].append(f"Column {col} contains missing values")
    
    return validation_results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', 'A', 'C', 'B', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    success = clean_csv_data('sample_data.csv', 'cleaned_data.csv', drop_na=False, fill_strategy='mean')
    
    if success:
        cleaned_df = pd.read_csv('cleaned_data.csv')
        validation = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
        print(f"Validation results: {validation}")
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        columns_to_check (list, optional): Specific columns to check for duplicates.
            If None, checks all columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Normalize string columns
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        # Convert to string, strip whitespace, and convert to lowercase
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
        
        # Replace empty strings with NaN
        cleaned_df[col] = cleaned_df[col].replace(['', 'nan', 'none'], np.nan)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate that DataFrame meets basic quality requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of columns that must be present
    
    Returns:
        dict: Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['all_required_columns_present'] = len(missing_columns) == 0
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', '   alice   ', 'BOB'],
        'email': ['john@example.com', 'jane@example.com', 'john@example.com', 'alice@example.com', None],
        'age': [25, 30, 25, 28, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned data
    validation = validate_data(cleaned, required_columns=['name', 'email'])
    print("Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")import pandas as pd
import numpy as np

def clean_csv_data(file_path, missing_strategy='mean', remove_duplicates=True):
    """
    Load and clean CSV data by handling missing values and duplicates.
    
    Args:
        file_path (str): Path to the CSV file.
        missing_strategy (str): Strategy for handling missing values.
                                Options: 'mean', 'median', 'drop', 'zero'.
        remove_duplicates (bool): Whether to remove duplicate rows.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_shape = df.shape
    
    if remove_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")
    
    if df.isnull().sum().any():
        print("Handling missing values...")
        for column in df.columns:
            if df[column].dtype in [np.float64, np.int64]:
                if missing_strategy == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif missing_strategy == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
                elif missing_strategy == 'zero':
                    df[column].fillna(0, inplace=True)
                elif missing_strategy == 'drop':
                    df = df.dropna(subset=[column])
            else:
                df[column].fillna('Unknown', inplace=True)
    
    print(f"Data cleaned. Original shape: {original_shape}, New shape: {df.shape}")
    return df

def validate_numeric_columns(df, columns):
    """
    Validate that specified columns contain only numeric values.
    
    Args:
        df (pandas.DataFrame): DataFrame to validate.
        columns (list): List of column names to check.
    
    Returns:
        bool: True if all columns are numeric, False otherwise.
    """
    for column in columns:
        if column not in df.columns:
            print(f"Column '{column}' not found in DataFrame.")
            return False
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            print(f"Column '{column}' is not numeric.")
            return False
    
    return True

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a column using the IQR method.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data.
        column (str): Column name to process.
        multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
        pandas.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric.")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    outliers_removed = len(df) - len(filtered_df)
    print(f"Removed {outliers_removed} outliers from column '{column}'.")
    
    return filtered_df