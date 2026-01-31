import pandas as pd

def clean_dataframe(df, fill_strategy='drop', column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    fill_strategy (str): Strategy for handling missing values - 'drop', 'fill_mean', 'fill_median', 'fill_mode'
    column_case (str): Target case for column names - 'lower', 'upper', 'title'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_strategy == 'fill_mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_strategy == 'fill_median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_strategy == 'fill_mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    
    # Standardize column names
    if column_case == 'lower':
        cleaned_df.columns = cleaned_df.columns.str.lower()
    elif column_case == 'upper':
        cleaned_df.columns = cleaned_df.columns.str.upper()
    elif column_case == 'title':
        cleaned_df.columns = cleaned_df.columns.str.title()
    
    # Remove any leading/trailing whitespace from column names
    cleaned_df.columns = cleaned_df.columns.str.strip()
    
    # Replace spaces with underscores in column names
    cleaned_df.columns = cleaned_df.columns.str.replace(' ', '_')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    min_rows (int): Minimum number of rows required
    
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
    
    return True, "DataFrame is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, 30, None, 35],
        'Score': [85.5, 92.0, 78.5, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned = clean_dataframe(df, fill_strategy='fill_mean', column_case='lower')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned, required_columns=['name', 'age'], min_rows=3)
    print(f"Validation: {is_valid}")
    print(f"Message: {message}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using Min-Max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            
            if col_max != col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def z_score_normalize(dataframe, columns=None):
    """
    Normalize specified columns using Z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with Z-score normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            
            if std_val > 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
            else:
                normalized_df[col] = 0
    
    return normalized_df

def clean_dataset(dataframe, outlier_columns=None, normalize_method='minmax', normalize_columns=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame
        outlier_columns: list of columns for outlier removal (default: all numeric columns)
        normalize_method: 'minmax' or 'zscore' (default: 'minmax')
        normalize_columns: list of columns to normalize (default: all numeric columns)
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = dataframe.copy()
    
    if outlier_columns is None:
        outlier_columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in outlier_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    if normalize_method == 'minmax':
        cleaned_df = normalize_minmax(cleaned_df, normalize_columns)
    elif normalize_method == 'zscore':
        cleaned_df = z_score_normalize(cleaned_df, normalize_columns)
    else:
        raise ValueError("Normalize method must be 'minmax' or 'zscore'")
    
    return cleaned_df

def get_data_summary(dataframe):
    """
    Generate statistical summary of the DataFrame.
    
    Args:
        dataframe: pandas DataFrame
    
    Returns:
        Dictionary containing statistical summary
    """
    summary = {
        'shape': dataframe.shape,
        'columns': dataframe.columns.tolist(),
        'dtypes': dataframe.dtypes.to_dict(),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': dataframe[col].mean(),
            'std': dataframe[col].std(),
            'min': dataframe[col].min(),
            '25%': dataframe[col].quantile(0.25),
            '50%': dataframe[col].quantile(0.50),
            '75%': dataframe[col].quantile(0.75),
            'max': dataframe[col].max(),
            'skewness': dataframe[col].skew(),
            'kurtosis': dataframe[col].kurtosis()
        }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset shape:", df.shape)
    
    cleaned_df = clean_dataset(df, normalize_method='zscore')
    print("Cleaned dataset shape:", cleaned_df.shape)
    
    summary = get_data_summary(cleaned_df)
    print("\nData summary generated successfully")