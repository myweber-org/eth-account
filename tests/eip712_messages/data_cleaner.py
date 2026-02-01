
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers using IQR method
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr', normalize=True):
    """
    Main cleaning function for datasets
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize:
            cleaned_df = normalize_minmax(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, min_rows=10):
    """
    Validate dataset structure and content
    """
    if len(df) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataset to file
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000)
    })
    
    # Add some outliers
    sample_data.loc[10, 'feature1'] = 500
    sample_data.loc[20, 'feature2'] = 1000
    
    cleaned = clean_dataset(sample_data, ['feature1', 'feature2', 'feature3'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    
    if validate_data(cleaned, ['feature1', 'feature2', 'feature3']):
        save_cleaned_data(cleaned, 'cleaned_data.csv')import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    original_shape = df.shape
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    removed_count = original_shape[0] - cleaned_df.shape[0]
    
    print(f"Removed {removed_count} duplicate rows")
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): 'mean', 'median', 'mode', or 'drop'
        columns (list, optional): Specific columns to clean
    
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_cleaned = df.copy()
    
    for col in columns:
        if col in df.columns:
            if strategy == 'mean':
                df_cleaned[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df_cleaned[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df_cleaned[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_cleaned = df_cleaned.dropna(subset=[col])
    
    return df_cleaned

def validate_data(df, rules):
    """
    Validate DataFrame against specified rules.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        rules (dict): Validation rules
    
    Returns:
        dict: Validation results
    """
    results = {
        'passed': True,
        'errors': [],
        'warnings': []
    }
    
    for column, rule in rules.items():
        if column in df.columns:
            if 'min' in rule:
                if df[column].min() < rule['min']:
                    results['passed'] = False
                    results['errors'].append(f"{column}: Value below minimum {rule['min']}")
            
            if 'max' in rule:
                if df[column].max() > rule['max']:
                    results['passed'] = False
                    results['errors'].append(f"{column}: Value above maximum {rule['max']}")
            
            if 'allowed_values' in rule:
                invalid_values = df[~df[column].isin(rule['allowed_values'])][column].unique()
                if len(invalid_values) > 0:
                    results['warnings'].append(f"{column}: Contains unexpected values: {invalid_values}")
    
    return results

def main():
    # Example usage
    data = {
        'id': [1, 2, 3, 1, 4, 5, 3],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David', 'Eve', 'Charlie'],
        'age': [25, 30, 35, 25, 28, np.nan, 35],
        'score': [85, 92, 78, 85, 95, 88, 78]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Remove duplicates
    df_clean = remove_duplicates(df, subset=['id', 'name'])
    
    # Clean missing values
    df_clean = clean_missing_values(df_clean, strategy='mean', columns=['age'])
    
    print("\nCleaned DataFrame:")
    print(df_clean)
    
    # Validate data
    validation_rules = {
        'age': {'min': 18, 'max': 100},
        'score': {'min': 0, 'max': 100}
    }
    
    validation_results = validate_data(df_clean, validation_rules)
    print("\nValidation Results:")
    print(validation_results)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import sys

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

def clean_data(df):
    original_rows = len(df)
    
    df.replace(['', 'NA', 'N/A', 'null', 'NULL'], np.nan, inplace=True)
    
    df.drop_duplicates(inplace=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    cleaned_rows = len(df)
    print(f"Data cleaning complete. Removed {original_rows - cleaned_rows} duplicate rows.")
    return df

def save_cleaned_data(df, output_path):
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Starting data cleaning process...")
    df = load_csv(input_file)
    df_cleaned = clean_data(df)
    save_cleaned_data(df_cleaned, output_file)
    print("Data cleaning process completed successfully.")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
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
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, max, and IQR.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'count': df[column].count(),
        'mean': df[column].mean(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'Q1': df[column].quantile(0.25),
        'Q3': df[column].quantile(0.75),
        'IQR': df[column].quantile(0.75) - df[column].quantile(0.25)
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary statistics:")
    original_stats = calculate_summary_statistics(df, 'value')
    for key, value in original_stats.items():
        print(f"  {key}: {value:.2f}")
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary statistics:")
    cleaned_stats = calculate_summary_statistics(cleaned_df, 'value')
    for key, value in cleaned_stats.items():
        print(f"  {key}: {value:.2f}")
    
    removed_count = df.shape[0] - cleaned_df.shape[0]
    print(f"\nRemoved {removed_count} outliers ({removed_count/df.shape[0]*100:.1f}% of data)")

if __name__ == "__main__":
    example_usage()