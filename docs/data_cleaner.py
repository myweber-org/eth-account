
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Remove outliers using z-score
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
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

def normalize_column(dataframe, column, method='minmax'):
    """
    Normalize a column in DataFrame.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to normalize
        method: normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column added as new column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = dataframe.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        
        if max_val == min_val:
            df_copy[f'{column}_normalized'] = 0.5
        else:
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        
        if std_val == 0:
            df_copy[f'{column}_normalized'] = 0
        else:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def clean_dataset(dataframe, numeric_columns=None, outlier_multiplier=1.5, normalize=True):
    """
    Comprehensive dataset cleaning function.
    
    Args:
        dataframe: pandas DataFrame to clean
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_multiplier: IQR multiplier for outlier removal
        normalize: whether to normalize columns
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    # Remove outliers from each numeric column
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_multiplier)
    
    # Normalize columns if requested
    if normalize:
        for column in numeric_columns:
            if column in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, column, method='minmax')
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def get_statistics(dataframe, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        dataframe: pandas DataFrame
        column: column name
    
    Returns:
        Dictionary of statistics
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats_dict = {
        'mean': dataframe[column].mean(),
        'median': dataframe[column].median(),
        'std': dataframe[column].std(),
        'min': dataframe[column].min(),
        'max': dataframe[column].max(),
        'count': dataframe[column].count(),
        'missing': dataframe[column].isnull().sum()
    }
    
    return stats_dict

# Example usage demonstration
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some outliers
    df.loc[95, 'value'] = 300
    df.loc[96, 'value'] = -50
    
    print("Original data shape:", df.shape)
    print("Original statistics:", get_statistics(df, 'value'))
    
    # Clean the data
    cleaned = clean_dataset(df, numeric_columns=['value'])
    
    print("\nCleaned data shape:", cleaned.shape)
    print("Cleaned statistics:", get_statistics(cleaned, 'value'))
    
    # Show normalized values
    if 'value_normalized' in cleaned.columns:
        print("\nNormalized value range: [{:.3f}, {:.3f}]".format(
            cleaned['value_normalized'].min(),
            cleaned['value_normalized'].max()
        ))