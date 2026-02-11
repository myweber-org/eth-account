
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    mean = df[column].mean()
    std = df[column].std()
    df[column] = (df[column] - mean) / std
    return df

def clean_dataset(input_path, output_path):
    df = load_data(input_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_column(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    cleaned_df = clean_dataset(input_file, output_file)import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe: Input DataFrame
        subset: Column label or sequence of labels to consider for identifying duplicates
        keep: Determines which duplicates to keep ('first', 'last', or False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by removing non-numeric characters and converting to float.
    
    Args:
        dataframe: Input DataFrame
        columns: List of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for column in columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column].astype(str).str.replace(r'[^\d.-]', '', regex=True),
                errors='coerce'
            )
    
    return dataframe

def validate_dataframe(dataframe, required_columns):
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        dataframe: Input DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Boolean indicating if validation passed
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    return True

def process_dataframe(dataframe, cleaning_config):
    """
    Main function to process DataFrame with multiple cleaning operations.
    
    Args:
        dataframe: Input DataFrame
        cleaning_config: Dictionary containing cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    if not validate_dataframe(dataframe, cleaning_config.get('required_columns', [])):
        raise ValueError("DataFrame validation failed")
    
    df = dataframe.copy()
    
    if 'remove_duplicates' in cleaning_config:
        config = cleaning_config['remove_duplicates']
        df = remove_duplicates(df, **config)
    
    if 'clean_numeric' in cleaning_config:
        df = clean_numeric_columns(df, cleaning_config['clean_numeric'])
    
    return df
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop').
    columns (list): List of columns to process. If None, process all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == 'mean':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == 'median':
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == 'mode':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to normalize.
    method (str): Normalization method ('minmax' or 'zscore').
    
    Returns:
    pd.DataFrame: DataFrame with normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df_normalized

def get_data_summary(df):
    """
    Generate a summary statistics DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: Summary statistics for each numeric column.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary_data = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            summary_data.append({
                'column': col,
                'count': len(col_data),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                '25%': col_data.quantile(0.25),
                '50%': col_data.quantile(0.50),
                '75%': col_data.quantile(0.75),
                'max': col_data.max(),
                'missing': df[col].isna().sum()
            })
    
    return pd.DataFrame(summary_data)