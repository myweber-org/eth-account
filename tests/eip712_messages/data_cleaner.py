import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for outlier detection ('iqr', 'zscore')
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in df_clean.columns]
    
    # Handle missing values
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            if missing_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif missing_strategy == 'median':
                fill_value = df_clean[col].median()
            elif missing_strategy == 'mode':
                fill_value = df_clean[col].mode()[0]
            elif missing_strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown missing strategy: {missing_strategy}")
            
            df_clean[col] = df_clean[col].fillna(fill_value)
    
    # Handle outliers
    for col in numeric_cols:
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif outlier_method == 'zscore':
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            
            # Identify outliers beyond 3 standard deviations
            z_scores = np.abs((df_clean[col] - mean_val) / std_val)
            outlier_mask = z_scores > 3
            
            # Replace outliers with median
            if outlier_mask.any():
                median_val = df_clean[col].median()
                df_clean.loc[outlier_mask, col] = median_val
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): Columns that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def get_data_summary(df):
    """
    Generate a comprehensive summary of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Summary statistics and metadata
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skewness': df[col].skew()
        }
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_stats'][col] = {
            'unique_count': df[col].nunique(),
            'top_value': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'top_frequency': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        }
    
    return summary

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, 8, 9],
        'C': ['x', 'y', 'x', 'z', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean the data
    df_clean = clean_dataframe(df, missing_strategy='median', outlier_method='iqr')
    print("Cleaned DataFrame:")
    print(df_clean)
    print()
    
    # Validate
    is_valid, message = validate_dataframe(df_clean, required_columns=['A', 'B', 'C'])
    print(f"Validation: {is_valid} - {message}")
    print()
    
    # Get summary
    summary = get_data_summary(df_clean)
    print("Data Summary:")
    print(f"Shape: {summary['shape']}")
    print(f"Numeric Stats for column A: {summary['numeric_stats']['A']}")