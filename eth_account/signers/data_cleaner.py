import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, normalize_columns=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    normalize_columns (bool): Whether to normalize column names
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_rows = initial_rows - len(cleaned_df)
        print(f"Removed {removed_rows} duplicate rows")
    
    if normalize_columns:
        cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
        print("Column names normalized")
    
    return cleaned_df

def validate_data(df, required_columns=None, check_missing=True):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    check_missing (bool): Whether to check for missing values
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': []
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing columns: {missing_columns}")
    
    if check_missing:
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            validation_results['issues'].append(f"Found {missing_values} missing values")
    
    return validation_results

def sample_data(df, sample_size=1000, random_state=42):
    """
    Create a random sample from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    sample_size (int): Number of rows to sample
    random_state (int): Random seed for reproducibility
    
    Returns:
    pd.DataFrame: Sampled DataFrame
    """
    if len(df) <= sample_size:
        return df
    
    return df.sample(n=sample_size, random_state=random_state)

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'David'],
        'Age': [25, 30, 25, 35, 40],
        'City': ['New York', 'London', 'New York', 'Paris', 'Tokyo']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_data(cleaned, required_columns=['name', 'age'])
    print(f"\nValidation results: {validation}")
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list, optional): List of numeric columns to clean.
            If None, uses all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 50, 10)
        ]),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original dataset shape: {df.shape}")
    
    cleaned_df = clean_dataset(df, ['value'])
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    
    print("\nSummary statistics:")
    print(f"Original - Mean: {df['value'].mean():.2f}, Std: {df['value'].std():.2f}")
    print(f"Cleaned - Mean: {cleaned_df['value'].mean():.2f}, Std: {cleaned_df['value'].std():.2f}")