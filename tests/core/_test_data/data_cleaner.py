
import pandas as pd
import numpy as np

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by standardizing columns, removing duplicates,
    and handling missing values.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def standardize_strings(df, columns):
    """
    Standardize string columns by converting to lowercase and stripping whitespace.
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.lower().str.strip()
    return df_copy

def validate_data(df, required_columns, date_columns=None):
    """
    Validate DataFrame structure and data types.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if date_columns:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

def process_data_file(input_path, output_path, config):
    """
    Main function to process a data file with cleaning operations.
    """
    try:
        df = pd.read_csv(input_path)
        
        df = clean_dataframe(
            df,
            column_mapping=config.get('column_mapping'),
            drop_duplicates=config.get('drop_duplicates', True),
            fill_missing=config.get('fill_missing', True)
        )
        
        if config.get('standardize_columns'):
            df = standardize_strings(df, config['standardize_columns'])
        
        if config.get('required_columns'):
            df = validate_data(
                df,
                config['required_columns'],
                config.get('date_columns')
            )
        
        df.to_csv(output_path, index=False)
        print(f"Data processed successfully. Output saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return False

if __name__ == "__main__":
    config = {
        'column_mapping': {'old_name': 'new_name'},
        'drop_duplicates': True,
        'fill_missing': True,
        'standardize_columns': ['name', 'category'],
        'required_columns': ['id', 'name', 'value'],
        'date_columns': ['date']
    }
    
    process_data_file('input_data.csv', 'cleaned_data.csv', config)
import pandas as pd
import numpy as np

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
    
    outliers_removed = len(df) - len(filtered_df)
    if outliers_removed > 0:
        print(f"Removed {outliers_removed} outliers from column '{column}'")
    
    return filtered_df.reset_index(drop=True)

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame containing only the outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return outliers

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Error processing column '{col}': {e}")
    
    return cleaned_df
import numpy as np
import pandas as pd

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
    
    return filtered_df.reset_index(drop=True)

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

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 15, 90),
            np.random.normal(300, 50, 10)
        ])
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_summary_statistics(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask where True indicates outliers.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, columns, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    Returns cleaned DataFrame.
    """
    clean_data = data.copy()
    outlier_mask = pd.Series([False] * len(data))
    
    for col in columns:
        if col in data.columns:
            col_outliers = detect_outliers_iqr(data, col, threshold)
            outlier_mask = outlier_mask | col_outliers
    
    return clean_data[~outlier_mask]

def normalize_minmax(data, columns):
    """
    Apply min-max normalization to specified columns.
    Returns DataFrame with normalized values.
    """
    normalized_data = data.copy()
    
    for col in columns:
        if col in data.columns:
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            
            if max_val > min_val:
                normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
    
    return normalized_data

def standardize_zscore(data, columns):
    """
    Apply z-score standardization to specified columns.
    Returns DataFrame with standardized values.
    """
    standardized_data = data.copy()
    
    for col in columns:
        if col in data.columns:
            mean_val = standardized_data[col].mean()
            std_val = standardized_data[col].std()
            
            if std_val > 0:
                standardized_data[col] = (standardized_data[col] - mean_val) / std_val
    
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = data.columns
    
    cleaned_data = data.copy()
    
    if strategy == 'drop':
        return cleaned_data.dropna(subset=columns)
    
    for col in columns:
        if col in cleaned_data.columns:
            if cleaned_data[col].isnull().any():
                if strategy == 'mean':
                    fill_value = cleaned_data[col].mean()
                elif strategy == 'median':
                    fill_value = cleaned_data[col].median()
                elif strategy == 'mode':
                    fill_value = cleaned_data[col].mode()[0]
                else:
                    continue
                
                cleaned_data[col] = cleaned_data[col].fillna(fill_value)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate comprehensive summary statistics for DataFrame.
    """
    summary = {
        'shape': data.shape,
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_stats': data.describe().to_dict() if data.select_dtypes(include=[np.number]).shape[1] > 0 else {},
        'categorical_stats': {col: data[col].value_counts().to_dict() 
                             for col in data.select_dtypes(include=['object']).columns}
    }
    
    return summary