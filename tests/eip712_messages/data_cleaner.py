
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