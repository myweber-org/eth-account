
import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_method='iqr', threshold=1.5):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for outlier detection ('iqr', 'zscore')
    threshold (float): Threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    
    cleaned_df = df.copy()
    
    # Handle missing values
    if strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif strategy == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Remove outliers
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
            cleaned_df = cleaned_df[mask]
        
        elif outlier_method == 'zscore':
            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
            cleaned_df = cleaned_df[z_scores < threshold]
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns=None, min_rows=10):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, message)
    """
    
    if len(df) < min_rows:
        return False, f"Dataset has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if df.isnull().sum().sum() > len(df) * 0.5:
        return False, "Dataset contains more than 50% missing values"
    
    return True, "Dataset validation passed"

def normalize_data(df, method='minmax'):
    """
    Normalize numeric columns in the dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    method (str): Normalization method ('minmax', 'standard')
    
    Returns:
    pd.DataFrame: Normalized dataframe
    """
    
    normalized_df = df.copy()
    numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == 'minmax':
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        
        elif method == 'standard':
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            if std_val != 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    
    return normalized_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'feature1': [1, 2, np.nan, 4, 5, 100],
        'feature2': [10, 20, 30, np.nan, 50, 60],
        'category': ['A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n")
    
    # Clean the dataset
    cleaned = clean_dataset(df, strategy='mean', outlier_method='iqr')
    print("Cleaned dataset:")
    print(cleaned)
    print("\n")
    
    # Validate the cleaned dataset
    is_valid, message = validate_data(cleaned, min_rows=3)
    print(f"Validation: {is_valid} - {message}")
    print("\n")
    
    # Normalize the cleaned dataset
    normalized = normalize_data(cleaned, method='minmax')
    print("Normalized dataset:")
    print(normalized)