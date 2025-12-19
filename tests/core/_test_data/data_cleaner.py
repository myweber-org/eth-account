
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to remove duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. 
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Dataset is valid"
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file"""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column, method='minmax'):
    """Normalize column using specified method"""
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        df[column] = (df[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        df[column] = stats.zscore(df[column])
    return df

def clean_dataset(df, numeric_columns):
    """Main cleaning function"""
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            cleaned_df = normalize_column(cleaned_df, column, method='zscore')
    
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

def save_cleaned_data(df, output_path):
    """Save cleaned dataset to CSV"""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    raw_data = load_dataset(input_file)
    numeric_cols = raw_data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = clean_dataset(raw_data, numeric_cols)
    save_cleaned_data(cleaned_data, output_file)
    
    print(f"Original shape: {raw_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(f"Data saved to: {output_file}")