import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
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
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
        print("Filled missing values with mode")
    
    print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation['is_valid'] = False
        validation['errors'].append("Input is not a pandas DataFrame")
        return validation
    
    if df.empty:
        validation['warnings'].append("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation['is_valid'] = False
            validation['errors'].append(f"Missing required columns: {missing_cols}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            validation['warnings'].append(f"Column '{col}' contains missing values")
    
    return validation

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 5],
        'B': [10, 20, 20, None, 50, 60],
        'C': ['x', 'y', 'y', 'z', 'z', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print("\nValidation results:", validation)
import pandas as pd

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing rows with null values
    and standardizing column names to lowercase with underscores.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Standardize column names
    df_cleaned.columns = (
        df_cleaned.columns
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )
    
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate that the DataFrame meets basic requirements.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def get_numeric_columns(df):
    """
    Return list of numeric columns in the DataFrame.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return numeric_cols

def get_categorical_columns(df):
    """
    Return list of categorical columns in the DataFrame.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return categorical_cols
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    """
    Normalize a column using min-max scaling to range [0, 1].
    """
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        return df[column].apply(lambda x: 0.5)
    
    return (df[column] - min_val) / (max_val - min_val)

def standardize_zscore(df, column):
    """
    Standardize a column using z-score normalization.
    """
    mean_val = df[column].mean()
    std_val = df[column].std()
    
    if std_val == 0:
        return df[column].apply(lambda x: 0)
    
    return (df[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_multiplier=1.5, normalization_method='minmax'):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_multiplier)
            
            if normalization_method == 'minmax':
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
            elif normalization_method == 'zscore':
                cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df, required_columns):
    """
    Validate that DataFrame contains all required columns and no null values.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    null_counts = df[required_columns].isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0].index.tolist()
    
    if columns_with_nulls:
        print(f"Warning: Columns with null values: {columns_with_nulls}")
    
    return True

def sample_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    sample_data = {
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original shape: {df.shape}")
    
    numeric_cols = ['feature1', 'feature2']
    cleaned_df = clean_dataset(df, numeric_cols, normalization_method='zscore')
    
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Cleaned statistics:\n{cleaned_df[numeric_cols].describe()}")
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = sample_usage()