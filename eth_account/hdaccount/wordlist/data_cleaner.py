import pandas as pd
import numpy as np
import sys

def clean_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Original shape: {df.shape}")
        
        df_cleaned = df.copy()
        
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"After removing duplicates: {df_cleaned.shape}")
        
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        
        text_columns = df_cleaned.select_dtypes(include=['object']).columns
        for col in text_columns:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col].fillna('Unknown', inplace=True)
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        print(f"Final shape: {df_cleaned.shape}")
        
        return True
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file.csv> <output_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = clean_csv(input_file, output_file)
    sys.exit(0 if success else 1)
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
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

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
    return data[mask]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame or Series
        column: column name to normalize
    
    Returns:
        Normalized data
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data
    
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return series * 0
    
    return (series - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame or Series
        column: column name to normalize
    
    Returns:
        Standardized data
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data
    
    mean_val = series.mean()
    std_val = series.std()
    
    if std_val == 0:
        return series * 0
    
    return (series - mean_val) / std_val

def clean_dataset(data, outlier_method='iqr', normalize_method='minmax', outlier_params=None, normalize_params=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
        outlier_params: dictionary of parameters for outlier removal
        normalize_params: dictionary of parameters for normalization
    
    Returns:
        Cleaned DataFrame
    """
    if outlier_params is None:
        outlier_params = {}
    if normalize_params is None:
        normalize_params = {}
    
    cleaned_data = data.copy()
    
    numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
    
    if outlier_method:
        for column in numeric_columns:
            try:
                if outlier_method == 'iqr':
                    cleaned_data = remove_outliers_iqr(cleaned_data, column, **outlier_params)
                elif outlier_method == 'zscore':
                    cleaned_data = remove_outliers_zscore(cleaned_data, column, **outlier_params)
            except Exception as e:
                print(f"Warning: Could not remove outliers from {column}: {e}")
    
    if normalize_method:
        for column in numeric_columns:
            try:
                if normalize_method == 'minmax':
                    cleaned_data[column] = normalize_minmax(cleaned_data, column, **normalize_params)
                elif normalize_method == 'zscore':
                    cleaned_data[column] = normalize_zscore(cleaned_data, column, **normalize_params)
            except Exception as e:
                print(f"Warning: Could not normalize {column}: {e}")
    
    return cleaned_data

def validate_data(data, check_missing=True, check_duplicates=True, check_infinite=True):
    """
    Validate data quality.
    
    Args:
        data: pandas DataFrame
        check_missing: check for missing values
        check_duplicates: check for duplicate rows
        check_infinite: check for infinite values
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    if check_missing:
        missing_counts = data.isnull().sum()
        missing_percentage = (missing_counts / len(data)) * 100
        validation_results['missing_values'] = {
            'counts': missing_counts.to_dict(),
            'percentage': missing_percentage.to_dict()
        }
    
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        validation_results['duplicates'] = {
            'count': duplicate_count,
            'percentage': (duplicate_count / len(data)) * 100
        }
    
    if check_infinite:
        numeric_data = data.select_dtypes(include=[np.number])
        infinite_mask = np.isinf(numeric_data)
        infinite_counts = infinite_mask.sum()
        validation_results['infinite_values'] = infinite_counts.to_dict()
    
    return validation_results