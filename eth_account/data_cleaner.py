
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
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 21),
        'value': [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                  21, 22, 23, 24, 25, 100, 110, 120, 130, 140]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nOriginal statistics:")
    print(calculate_basic_stats(df, 'value'))
    
    cleaned_df = clean_numeric_data(df, ['value'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned statistics:")
    print(calculate_basic_stats(cleaned_df, 'value'))import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    
    df_cleaned = df.copy()
    
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df_cleaned[col].dropna()))
        outliers = z_scores > 3
        df_cleaned.loc[outliers, col] = np.nan
        print(f"Removed {outliers.sum()} outliers from {col}")
    
    df_cleaned = df_cleaned.dropna(subset=numeric_cols)
    print(f"Shape after outlier removal: {df_cleaned.shape}")
    
    for col in numeric_cols:
        col_min = df_cleaned[col].min()
        col_max = df_cleaned[col].max()
        if col_max > col_min:
            df_cleaned[col] = (df_cleaned[col] - col_min) / (col_max - col_min)
        else:
            df_cleaned[col] = 0
    
    print("Data normalization completed")
    return df_cleaned

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = load_and_clean_data(input_file)
        save_cleaned_data(cleaned_df, output_file)
        print("Data cleaning pipeline executed successfully")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame containing the data
        column: string name of the column to clean
    
    Returns:
        pandas DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        data: pandas DataFrame
        column: string name of the column
    
    Returns:
        dictionary containing mean, median, std, min, max
    """
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column])
    }
    
    return stats

def clean_dataset(data, numeric_columns):
    """
    Clean dataset by removing outliers from all specified numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of column names to clean
    
    Returns:
        cleaned pandas DataFrame
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data
import pandas as pd
import numpy as np

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

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_minmax(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Original shape: {pd.read_csv(input_file).shape}")
    print(f"Cleaned shape: {df.shape}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    # Handle missing values
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values")
        elif fill_missing == 'mean':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
            print("Filled missing values with column means")
        elif fill_missing == 'median':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
            print("Filled missing values with column medians")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
            print("Filled missing values with column modes")
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    print(f"Cleaning complete. Final dataset shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if len(df) < min_rows:
        print(f"Validation failed: Dataset has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        if inf_count > 0:
            print(f"Validation warning: Found {inf_count} infinite values in numeric columns")
    
    print("Dataset validation passed")
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', None],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    
    print("\n" + "="*50 + "\n")
    print("Cleaned dataset:")
    print(cleaned_df)
    
    # Validate the cleaned dataset
    print("\n" + "="*50 + "\n")
    validation_passed = validate_dataset(cleaned_df, required_columns=['id', 'name', 'age'], min_rows=3)
    
    if validation_passed:
        print("\nDataset is ready for analysis")
    else:
        print("\nDataset requires further cleaning")
import pandas as pd
import numpy as np

def clean_csv_data(file_path, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and optionally dropping columns.
    
    Parameters:
    file_path (str): Path to the CSV file.
    missing_strategy (str): Strategy for handling missing values. 
                            Options: 'mean', 'median', 'mode', 'drop'.
    columns_to_drop (list): List of column names to drop from the dataset.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
    
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy in ['mean', 'median', 'mode']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if missing_strategy == 'mean':
                fill_value = df[col].mean()
            elif missing_strategy == 'median':
                fill_value = df[col].median()
            elif missing_strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else np.nan
            
            df[col] = df[col].fillna(fill_value)
    
    return df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a specific column using the IQR method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    column (str): Column name to process.
    multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
    pandas.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a CSV file.
    
    Parameters:
    df (pandas.DataFrame): Cleaned DataFrame.
    output_path (str): Path to save the cleaned CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Column labels to consider for duplicates
    keep (str, optional): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name):
    """
    Clean a numeric column by converting to float and removing NaN values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_name (str): Name of the column to clean
    
    Returns:
    pd.DataFrame: DataFrame with cleaned column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df = df.dropna(subset=[column_name])
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list, optional): List of required column names
    
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

def clean_dataset(df, cleaning_steps=None):
    """
    Apply multiple cleaning steps to a dataset.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    cleaning_steps (list, optional): List of cleaning functions to apply
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if not validate_dataframe(df):
        raise ValueError("Invalid DataFrame provided")
    
    cleaned_df = df.copy()
    
    if cleaning_steps:
        for step in cleaning_steps:
            cleaned_df = step(cleaned_df)
    
    return cleaned_df
import pandas as pd

def clean_dataset(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): If True, drop rows with any null values.
        rename_columns (bool): If True, rename columns to lowercase with underscores.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df.columns = (
            cleaned_df.columns
            .str.lower()
            .str.replace(r'[^a-z0-9]+', '_', regex=True)
            .str.strip('_')
        )
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Dataset is valid"