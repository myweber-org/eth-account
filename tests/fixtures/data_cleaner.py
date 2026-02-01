import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values
                               'mean', 'median', 'drop', or 'zero'
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif missing_strategy == 'median':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        elif missing_strategy == 'drop':
            df = df.dropna(subset=numeric_cols)
        elif missing_strategy == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Remove outliers using IQR method for numeric columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        # Print cleaning summary
        print(f"Data cleaning completed:")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Final dataset has {len(df)} rows and {len(df.columns)} columns")
        print(f"  - Cleaned data saved to: {output_file}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Validate dataframe for common data quality issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns)
    }
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinite_count = 0
    for col in numeric_cols:
        infinite_count += np.isinf(df[col]).sum()
    validation_results['infinite_values'] = int(infinite_count)
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    # Clean the data
    cleaned_df = clean_csv_data(input_csv, output_csv, missing_strategy='median')
    
    if cleaned_df is not None:
        # Validate the cleaned data
        validation = validate_dataframe(cleaned_df)
        print("\nData Validation Results:")
        for key, value in validation.items():
            print(f"  {key}: {value}")
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
    
    return filtered_df

def clean_numeric_data(df, columns=None):
    """
    Clean numeric columns by removing outliers from specified columns or all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list, optional): List of column names to clean. If None, cleans all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

def get_cleaning_stats(original_df, cleaned_df):
    """
    Get statistics about the data cleaning process.
    
    Parameters:
    original_df (pd.DataFrame): Original DataFrame before cleaning
    cleaned_df (pd.DataFrame): DataFrame after cleaning
    
    Returns:
    dict: Dictionary containing cleaning statistics
    """
    stats = {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'removed_rows': len(original_df) - len(cleaned_df),
        'removed_percentage': ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500  # Add some outliers
    
    print("Original DataFrame shape:", df.shape)
    
    cleaned_df = clean_numeric_data(df, columns=['A', 'B'])
    
    print("Cleaned DataFrame shape:", cleaned_df.shape)
    
    stats = get_cleaning_stats(df, cleaned_df)
    print("Cleaning statistics:", stats)
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
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

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each cleaned column
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_basic_stats(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            
            all_stats[column] = stats
    
    return cleaned_df, all_stats

if __name__ == "__main__":
    sample_data = {
        'temperature': np.random.normal(25, 5, 1000),
        'humidity': np.random.normal(60, 15, 1000),
        'pressure': np.random.normal(1013, 10, 1000)
    }
    
    sample_data['temperature'][:50] = np.random.uniform(50, 100, 50)
    sample_data['humidity'][:30] = np.random.uniform(0, 10, 30)
    
    df = pd.DataFrame(sample_data)
    
    columns_to_process = ['temperature', 'humidity', 'pressure']
    
    cleaned_df, stats = clean_dataset(df, columns_to_process)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    
    for column, column_stats in stats.items():
        print(f"\nStatistics for {column}:")
        for stat_name, stat_value in column_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")
def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

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
    mask = z_scores < threshold
    
    valid_indices = data[column].dropna().index[mask]
    filtered_data = data.loc[valid_indices].copy()
    
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    col_data = data[column].copy()
    min_val = col_data.min()
    max_val = col_data.max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(col_data), index=col_data.index)
    
    normalized = (col_data - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    col_data = data[column].copy()
    mean_val = col_data.mean()
    std_val = col_data.std()
    
    if std_val == 0:
        return pd.Series([0] * len(col_data), index=col_data.index)
    
    standardized = (col_data - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr' or 'zscore' (default: 'iqr')
        normalize_method: 'minmax' or 'zscore' (default: 'minmax')
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
    
    return cleaned_data.reset_index(drop=True)

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame to validate
        required_columns: list of required column names
        allow_nan: whether to allow NaN values
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if not allow_nan and data.isnull().any().any():
        nan_cols = data.columns[data.isnull().any()].tolist()
        return False, f"NaN values found in columns: {nan_cols}"
    
    return True, "Data validation passed"import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """Remove outliers using IQR method."""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """Remove outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """Normalize data using Min-Max scaling."""
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_standard(data, column):
    """Normalize data using Standardization (Z-score normalization)."""
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr', normalize=False):
    """Main function to clean dataset by removing outliers and optionally normalizing."""
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize:
            cleaned_df = normalize_standard(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns, numeric_columns):
    """Validate dataset structure and data types."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column {col} must be numeric")
    
    return True

def get_summary_statistics(df, numeric_columns):
    """Generate summary statistics for numeric columns."""
    summary = {}
    for col in numeric_columns:
        if col in df.columns:
            summary[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': df[col].count(),
                'missing': df[col].isnull().sum()
            }
    return pd.DataFrame(summary).T

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 10 + 50,
        'score': np.random.randn(100) * 5 + 20
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset shape:", df.shape)
    
    numeric_cols = ['value', 'score']
    
    try:
        validate_data(df, ['id', 'value', 'score'], numeric_cols)
        cleaned_df = clean_dataset(df, numeric_cols, method='iqr', normalize=True)
        print("Cleaned dataset shape:", cleaned_df.shape)
        
        summary_stats = get_summary_statistics(cleaned_df, numeric_cols)
        print("\nSummary statistics:")
        print(summary_stats)
        
    except (ValueError, TypeError) as e:
        print(f"Data validation error: {e}")