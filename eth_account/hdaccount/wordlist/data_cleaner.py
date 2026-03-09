import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process, None for all numeric columns
    factor (float): Multiplier for IQR
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize, None for all numeric columns
    feature_range (tuple): Desired range of transformed data
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_norm = df.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        
        if col_max - col_min == 0:
            df_norm[col] = min_val
        else:
            df_norm[col] = min_val + (df_norm[col] - col_min) * (max_val - min_val) / (col_max - col_min)
    
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'constant')
    columns (list): List of column names to process, None for all columns
    
    Returns:
    pd.DataFrame: Dataframe with missing values handled
    """
    if columns is None:
        columns = df.columns
    
    df_filled = df.copy()
    
    for col in columns:
        if df_filled[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_filled[col].mean()
            elif strategy == 'median':
                fill_value = df_filled[col].median()
            elif strategy == 'mode':
                fill_value = df_filled[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_filled[col] = df_filled[col].fillna(fill_value)
    
    return df_filled

def z_score_normalize(df, columns=None, threshold=3):
    """
    Normalize data using z-score and optionally remove extreme outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process
    threshold (float): Z-score threshold for outlier removal
    
    Returns:
    pd.DataFrame: Normalized dataframe
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_z = df.copy()
    
    for col in columns:
        mean_val = df_z[col].mean()
        std_val = df_z[col].std()
        
        if std_val > 0:
            z_scores = np.abs((df_z[col] - mean_val) / std_val)
            df_z = df_z[z_scores <= threshold].copy()
            df_z[col] = (df_z[col] - df_z[col].mean()) / df_z[col].std()
    
    return df_z.reset_index(drop=True)

def create_cleaning_pipeline(df, steps):
    """
    Create a data cleaning pipeline with multiple steps.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    steps (list): List of cleaning functions and their arguments
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    for step in steps:
        func = step['function']
        kwargs = step.get('kwargs', {})
        cleaned_df = func(cleaned_df, **kwargs)
    
    return cleaned_df

# Example usage demonstration
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'feature3': np.random.uniform(0, 1, 100)
    }
    
    # Introduce some outliers and missing values
    sample_data['feature1'][[10, 20, 30]] = [500, -200, 1000]
    sample_data['feature2'][[15, 25]] = [np.nan, np.nan]
    
    df_sample = pd.DataFrame(sample_data)
    
    print("Original data shape:", df_sample.shape)
    print("Missing values:\n", df_sample.isnull().sum())
    
    # Define cleaning pipeline
    cleaning_steps = [
        {'function': handle_missing_values, 'kwargs': {'strategy': 'median'}},
        {'function': remove_outliers_iqr, 'kwargs': {'factor': 1.5}},
        {'function': normalize_minmax, 'kwargs': {'feature_range': (0, 1)}}
    ]
    
    # Apply cleaning pipeline
    cleaned_df = create_cleaning_pipeline(df_sample, cleaning_steps)
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Cleaned data statistics:")
    print(cleaned_df.describe())