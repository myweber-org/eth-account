
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
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

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].describe()
    
    summary.loc['variance'] = df[numeric_cols].var()
    summary.loc['skewness'] = df[numeric_cols].skew()
    summary.loc['kurtosis'] = df[numeric_cols].kurtosis()
    
    return summary

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df.dropna(subset=numeric_cols).reset_index(drop=True)
    
    df_filled = df.copy()
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to normalize (None for all numeric)
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: Normalized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'minmax':
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
        
        elif method == 'zscore':
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std > 0:
                df_normalized[col] = (df[col] - col_mean) / col_std
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return df_normalized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage demonstration
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    # Introduce some outliers and missing values
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[0, 'value'] = 500  # Outlier
    sample_df.loc[1, 'value'] = -100  # Outlier
    sample_df.loc[5:10, 'value'] = np.nan  # Missing values
    
    print("Original DataFrame shape:", sample_df.shape)
    print("Missing values:", sample_df.isnull().sum().sum())
    
    # Clean the data
    cleaned_df = handle_missing_values(sample_df, strategy='mean')
    cleaned_df = remove_outliers_iqr(cleaned_df, 'value')
    normalized_df = normalize_data(cleaned_df, columns=['value'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Normalized value statistics:")
    print(normalized_df['value'].describe())
    
    # Validate the result
    is_valid, message = validate_dataframe(normalized_df, min_rows=50)
    print(f"\nValidation: {message}")