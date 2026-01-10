
import pandas as pd

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
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        if fill_missing == 'mean':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        else:
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None
                cleaned_df[col] = cleaned_df[col].fillna(mode_val)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and basic data integrity.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (bool, str) indicating validation result and message.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', None, 'z']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid}, Message: {message}")
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def normalize_minmax(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    dataframe[column + '_normalized'] = (dataframe[column] - min_val) / (max_val - min_val)
    return dataframe

def clean_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_minmax(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Original shape: {pd.read_csv(input_path).shape}, Cleaned shape: {df.shape}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            original_count = len(df)
            df = remove_outliers_iqr(df, column)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column: {column}")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved to: {output_file}")
        print(f"Final dataset shape: {df.shape}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The name of the column to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_dataimport pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    column_mapping (dict): Optional dictionary to rename columns
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Strategy to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Rename columns if mapping is provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Drop duplicate rows if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    for column in cleaned_df.select_dtypes(include=[np.number]).columns:
        if cleaned_df[column].isnull().any():
            if fill_missing == 'mean':
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif fill_missing == 'mode':
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            elif fill_missing == 'drop':
                cleaned_df = cleaned_df.dropna(subset=[column])
    
    # For categorical columns, fill with most frequent value
    for column in cleaned_df.select_dtypes(include=['object']).columns:
        if cleaned_df[column].isnull().any():
            cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10.5, 20.3, np.nan, 30.1, 40.7, np.nan],
        'category': ['A', 'B', 'A', 'A', 'C', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    
    # Validate
    is_valid = validate_dataframe(cleaned, required_columns=['id', 'value'], min_rows=3)
    print(f"\nData validation passed: {is_valid}")
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a dataset by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for missing values - 'mean', 'median', 'mode', or 'drop'
    outlier_method (str): Method for outlier detection - 'iqr' or 'zscore'
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    else:
        numeric_cols = [col for col in columns if col in df_clean.columns]
    
    for col in numeric_cols:
        if missing_strategy != 'drop':
            if missing_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif missing_strategy == 'median':
                fill_value = df_clean[col].median()
            elif missing_strategy == 'mode':
                fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else np.nan
            else:
                fill_value = 0
            
            df_clean[col] = df_clean[col].fillna(fill_value)
        else:
            df_clean = df_clean.dropna(subset=[col])
    
    if outlier_method == 'iqr':
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean[col] = np.where(
                (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound),
                df_clean[col].median(),
                df_clean[col]
            )
    
    elif outlier_method == 'zscore':
        for col in numeric_cols:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            threshold = 3
            
            df_clean[col] = np.where(
                z_scores > threshold,
                df_clean[col].median(),
                df_clean[col]
            )
    
    return df_clean

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list): Columns to consider for duplicates
    keep (str): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame without duplicates
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_data(df, method='minmax', columns=None):
    """
    Normalize numeric data in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    method (str): Normalization method - 'minmax' or 'zscore'
    columns (list): Columns to normalize
    
    Returns:
    pd.DataFrame: Normalized DataFrame
    """
    df_norm = df.copy()
    
    if columns is None:
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
    else:
        numeric_cols = [col for col in columns if col in df_norm.columns]
    
    for col in numeric_cols:
        if method == 'minmax':
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val != min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val != 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm