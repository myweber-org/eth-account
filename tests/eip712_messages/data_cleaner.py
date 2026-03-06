
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary containing validation results.
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'data_types': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['is_valid'] = False
            validation_results['missing_columns'] = missing
    
    for column in df.columns:
        null_count = df[column].isnull().sum()
        if null_count > 0:
            validation_results['null_counts'][column] = null_count
        
        validation_results['data_types'][column] = str(df[column].dtype)
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': [10, 11, 12, 12, 13]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    print()
    
    validation = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print("Validation Results:")
    print(validation)
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame using specified strategy.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to apply cleaning to, None for all columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns
    
    for col in columns:
        if col in df_clean.columns:
            if strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                if not df_clean[col].mode().empty:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to check for outliers
    threshold (float): IQR multiplier threshold
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in [np.float64, np.int64]:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize numerical columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to standardize
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns
    """
    from sklearn.preprocessing import StandardScaler
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    scaler = StandardScaler()
    
    for col in columns:
        if col in df_standardized.columns and df_standardized[col].dtype in [np.float64, np.int64]:
            df_standardized[col] = scaler.fit_transform(df_standardized[[col]])
    
    return df_standardized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
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

def save_cleaned_data(df, output_path, index=False):
    """
    Save cleaned DataFrame to CSV file.
    
    Parameters:
    df (pd.DataFrame): DataFrame to save
    output_path (str): Path to save the CSV file
    index (bool): Whether to save index
    
    Returns:
    bool: True if save successful, False otherwise
    """
    try:
        df.to_csv(output_path, index=index)
        print(f"Data saved successfully to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False