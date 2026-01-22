
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def z_score_normalize(data, column):
    """
    Normalize data using Z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].copy()
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def min_max_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].copy()
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    
    if feature_range != (0, 1):
        new_min, new_max = feature_range
        normalized = normalized * (new_max - new_min) + new_min
    
    return normalized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for col in columns:
        if col not in data_copy.columns:
            continue
            
        if data_copy[col].isnull().any():
            if strategy == 'mean':
                fill_value = data_copy[col].mean()
            elif strategy == 'median':
                fill_value = data_copy[col].median()
            elif strategy == 'mode':
                fill_value = data_copy[col].mode()[0]
            elif strategy == 'drop':
                data_copy = data_copy.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_copy[col] = data_copy[col].fillna(fill_value)
    
    return data_copy

def validate_dataframe(data, required_columns=None):
    """
    Validate DataFrame structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.randint(1, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(df.index, 5), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 3), 'feature_b'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    
    cleaned_data = handle_missing_values(sample_data, strategy='mean')
    print("After handling missing values:", cleaned_data.shape)
    
    normalized_data = cleaned_data.copy()
    normalized_data['feature_a_norm'] = z_score_normalize(cleaned_data, 'feature_a')
    normalized_data['feature_b_norm'] = min_max_normalize(cleaned_data, 'feature_b')
    
    print("Data normalization completed")
    print("Normalized columns added:", ['feature_a_norm', 'feature_b_norm'])import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for duplicates.
            If None, checks all columns. Defaults to None.
        fill_missing (str or dict, optional): Method to fill missing values.
            Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            Defaults to 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_missing == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_missing == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_missing == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif isinstance(fill_missing, dict):
        cleaned_df = cleaned_df.fillna(fill_missing)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
        min_rows (int, optional): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Dataset is valid"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (fill with mean):")
    cleaned = clean_dataset(df, fill_missing='mean')
    print(cleaned)
    
    is_valid, message = validate_dataset(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid} - {message}")import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicate rows and
    filling missing numeric values with column mean.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values with column mean
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    
    # For non-numeric columns, fill with mode
    non_numeric_cols = df_cleaned.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        if df_cleaned[col].isnull().any():
            mode_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
            df_cleaned[col] = df_cleaned[col].fillna(mode_value)
    
    return df_cleaned

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve'],
        'age': [25, 30, 30, None, 35],
        'score': [85.5, 92.0, 92.0, 78.5, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    # Clean the data
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned_df, required_columns=['id', 'name', 'age'])
    print(f"\nValidation result: {is_valid}")
    print(f"Validation message: {message}")import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, fill_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output file
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(file_path)
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if fill_strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_strategy == 'mode':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        elif fill_strategy == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Fill non-numeric columns with empty string
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        df[non_numeric_cols] = df[non_numeric_cols].fillna('')
        
        # Save cleaned data if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
        
        # Print summary
        print(f"Data cleaning completed:")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Filled missing values using '{fill_strategy}' strategy")
        print(f"  - Final dataset shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pandas.DataFrame): Dataframe to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Validation failed: Dataframe is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    
    if inf_count > 0:
        print(f"Validation warning: Found {inf_count} infinite values")
    
    return True

# Example usage
if __name__ == "__main__":
    # Test the data cleaner
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', None, 'A', 'C', 'A']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv', 'mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, ['id', 'value', 'category'])
        print(f"Data validation: {'Passed' if is_valid else 'Failed'}")