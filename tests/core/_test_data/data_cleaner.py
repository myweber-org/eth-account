
import re
import json
from typing import Dict, Any, Optional, List

def remove_whitespace(text: str) -> str:
    """Remove extra whitespace from a string."""
    if not isinstance(text, str):
        return text
    return re.sub(r'\s+', ' ', text).strip()

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def normalize_phone(phone: str) -> Optional[str]:
    """Normalize phone number to digits only."""
    digits = re.sub(r'\D', '', phone)
    if len(digits) >= 10:
        return digits
    return None

def clean_json_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean all string values in a dictionary."""
    cleaned = {}
    for key, value in data.items():
        if isinstance(value, str):
            cleaned[key] = remove_whitespace(value)
        elif isinstance(value, dict):
            cleaned[key] = clean_json_data(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_json_data(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value
    return cleaned

def validate_user_data(user: Dict[str, Any]) -> List[str]:
    """Validate user data and return list of errors."""
    errors = []
    
    if 'name' not in user or not user['name'].strip():
        errors.append("Name is required")
    
    if 'email' in user:
        if not validate_email(user['email']):
            errors.append("Invalid email format")
    else:
        errors.append("Email is required")
    
    if 'phone' in user and user['phone']:
        normalized = normalize_phone(user['phone'])
        if not normalized:
            errors.append("Invalid phone number")
        elif len(normalized) > 15:
            errors.append("Phone number too long")
    
    return errors

def process_data_file(input_path: str, output_path: str) -> bool:
    """Process a JSON file, clean data, and save to output."""
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        cleaned_data = clean_json_data(data)
        
        with open(output_path, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def extract_domain(email: str) -> Optional[str]:
    """Extract domain from email address."""
    if not validate_email(email):
        return None
    return email.split('@')[1]

def generate_user_report(users: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate statistics report from user data."""
    total = len(users)
    valid_emails = sum(1 for u in users if 'email' in u and validate_email(u['email']))
    valid_phones = sum(1 for u in users if 'phone' in u and normalize_phone(u.get('phone', '')))
    
    domains = {}
    for user in users:
        if 'email' in user and validate_email(user['email']):
            domain = extract_domain(user['email'])
            if domain:
                domains[domain] = domains.get(domain, 0) + 1
    
    return {
        'total_users': total,
        'valid_emails': valid_emails,
        'valid_phones': valid_phones,
        'email_domains': domains,
        'completion_rate': (valid_emails / total * 100) if total > 0 else 0
    }
import numpy as np
import pandas as pd

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
    
    return filtered_df.reset_index(drop=True)

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df[column])
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 50, 5)
        ])
    }
    
    df = pd.DataFrame(data)
    
    print("Original data statistics:")
    original_stats = calculate_statistics(df, 'values')
    for key, value in original_stats.items():
        print(f"{key}: {value:.2f}")
    
    print(f"\nOriginal data shape: {df.shape}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    
    print("\nCleaned data statistics:")
    cleaned_stats = calculate_statistics(cleaned_df, 'values')
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.2f}")
    
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(f"Outliers removed: {len(df) - len(cleaned_df)}")

if __name__ == "__main__":
    example_usage()
import pandas as pd
import numpy as np
from typing import List, Optional

def clean_dataset(df: pd.DataFrame, 
                  drop_duplicates: bool = True,
                  columns_to_standardize: Optional[List[str]] = None,
                  date_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by handling duplicates, standardizing text,
    and parsing dates.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if columns_to_standardize:
        for col in columns_to_standardize:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    if date_columns:
        for col in date_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
    
    cleaned_df = cleaned_df.replace(['', 'nan', 'null', 'none'], np.nan)
    
    print(f"Cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame contains all required columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    print("DataFrame validation passed")
    return True

def sample_data(df: pd.DataFrame, sample_size: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Return a random sample from the DataFrame.
    """
    if len(df) <= sample_size:
        return df
    
    return df.sample(n=sample_size, random_state=random_state)
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
                            Options: 'mean', 'median', 'drop', 'fill_zero'.
    outlier_threshold (float): Number of standard deviations for outlier detection.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy == 'fill_zero':
        cleaned_df = cleaned_df.fillna(0)
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    # Handle outliers using Z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
        outlier_mask = z_scores > outlier_threshold
        
        if outlier_mask.any():
            # Replace outliers with column median
            col_median = cleaned_df[col].median()
            cleaned_df.loc[outlier_mask, col] = col_median
    
    # Reset index if rows were dropped
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

# Example usage function
def process_data_file(file_path, output_path=None):
    """
    Process a data file through the cleaning pipeline.
    
    Parameters:
    file_path (str): Path to input data file.
    output_path (str): Path to save cleaned data (optional).
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    try:
        # Read data file (supports CSV and Excel)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Validate data
        if not validate_dataframe(df):
            raise ValueError("Data validation failed")
        
        # Clean data
        cleaned_df = clean_dataset(df, missing_strategy='median', outlier_threshold=3)
        
        # Save cleaned data if output path provided
        if output_path:
            if output_path.endswith('.csv'):
                cleaned_df.to_csv(output_path, index=False)
            elif output_path.endswith(('.xls', '.xlsx')):
                cleaned_df.to_excel(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return cleaned_df
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_column(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def clean_data(input_file, output_file):
    df = load_data(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    clean_data("raw_data.csv", "cleaned_data.csv")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr', normalize=False):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    if normalize:
        for col in numeric_columns:
            cleaned_df = normalize_minmax(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, numeric_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    for col in numeric_columns:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    
    return df