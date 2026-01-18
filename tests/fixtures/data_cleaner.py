
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
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

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 14, 13, 12, 11, 10, 9, 8, 12, 13]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    stats = calculate_summary_statistics(cleaned_df, 'values')
    print("\nSummary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if subset is None:
        subset = dataframe.columns.tolist()
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if dataframe.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def clean_numeric_columns(dataframe, columns=None):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=['object']).columns
    
    cleaned_df = dataframe.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            try:
                cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
            except Exception as e:
                print(f"Error converting column {column}: {e}")
    
    return cleaned_df

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 3, 2, 4, 1],
        'name': ['Alice', 'Bob', 'Charlie', 'Bob', 'David', 'Alice'],
        'value': ['100', '200', '300', '200', '400', '100']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    if validate_dataframe(df, required_columns=['id', 'name']):
        cleaned_df = remove_duplicates(df, subset=['id', 'name'])
        cleaned_df = clean_numeric_columns(cleaned_df, columns=['value'])
        
        print("\nCleaned DataFrame:")
        print(cleaned_df)
        
        return cleaned_df

if __name__ == "__main__":
    main()import pandas as pd

def clean_dataset(df, column_names):
    """
    Remove duplicate rows and fill missing values with column mean.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing values with column mean for numeric columns
    for col in column_names:
        if df_cleaned[col].dtype in ['int64', 'float64']:
            df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
        else:
            df_cleaned[col].fillna('Unknown', inplace=True)
    
    return df_cleaned

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv('raw_data.csv')
    columns_to_clean = ['age', 'salary', 'department']
    cleaned_data = clean_dataset(data, columns_to_clean)
    save_cleaned_data(cleaned_data, 'cleaned_data.csv')import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    
    if len(data) != len(z_scores):
        valid_indices = data[column].dropna().index
        filtered_data = data.loc[valid_indices[filtered_indices]]
    else:
        filtered_data = data.iloc[filtered_indices]
    
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    min_val = data_copy[column].min()
    max_val = data_copy[column].max()
    
    if max_val == min_val:
        data_copy[f'{column}_normalized'] = 0.5
    else:
        data_copy[f'{column}_normalized'] = (data_copy[column] - min_val) / (max_val - min_val)
    
    return data_copy

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    mean_val = data_copy[column].mean()
    std_val = data_copy[column].std()
    
    if std_val == 0:
        data_copy[f'{column}_standardized'] = 0
    else:
        data_copy[f'{column}_standardized'] = (data_copy[column] - mean_val) / std_val
    
    return data_copy

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names to process
        outlier_method: 'iqr' or 'zscore' (default 'iqr')
        normalize_method: 'minmax' or 'zscore' (default 'minmax')
    
    Returns:
        Cleaned and normalized DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError("outlier_method must be 'iqr' or 'zscore'")
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if normalize_method == 'minmax':
            cleaned_data = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError("normalize_method must be 'minmax' or 'zscore'")
    
    return cleaned_data

def get_summary_statistics(data, numeric_columns):
    """
    Generate summary statistics for numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {}
    
    for column in numeric_columns:
        if column in data.columns:
            col_data = data[column].dropna()
            summary[column] = {
                'count': len(col_data),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                '25%': col_data.quantile(0.25),
                'median': col_data.median(),
                '75%': col_data.quantile(0.75),
                'max': col_data.max(),
                'missing': data[column].isnull().sum()
            }
    
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("\nOriginal summary statistics:")
    stats_original = get_summary_statistics(sample_data, ['feature1', 'feature2', 'feature3'])
    
    cleaned_data = clean_dataset(
        sample_data, 
        ['feature1', 'feature2', 'feature3'],
        outlier_method='iqr',
        normalize_method='minmax'
    )
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("\nCleaned summary statistics:")
    stats_cleaned = get_summary_statistics(cleaned_data, ['feature1', 'feature2', 'feature3'])
    
    print("\nProcessing complete.")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data