
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