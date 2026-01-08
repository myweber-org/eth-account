
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'zero')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_rows = initial_rows - len(cleaned_df)
        print(f"Removed {removed_rows} duplicate rows")
    
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].isnull().any():
                null_count = cleaned_df[column].isnull().sum()
                
                if cleaned_df[column].dtype in ['int64', 'float64']:
                    if fill_strategy == 'mean':
                        fill_value = cleaned_df[column].mean()
                    elif fill_strategy == 'median':
                        fill_value = cleaned_df[column].median()
                    elif fill_strategy == 'mode':
                        fill_value = cleaned_df[column].mode()[0]
                    elif fill_strategy == 'zero':
                        fill_value = 0
                    else:
                        fill_value = cleaned_df[column].mean()
                    
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    print(f"Filled {null_count} missing values in column '{column}' with {fill_strategy}: {fill_value}")
                
                elif cleaned_df[column].dtype == 'object':
                    mode_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown'
                    cleaned_df[column] = cleaned_df[column].fillna(mode_value)
                    print(f"Filled {null_count} missing values in column '{column}' with mode: {mode_value}")
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, validation_message)
    """
    validation_messages = []
    
    if len(df) < min_rows:
        validation_messages.append(f"Dataset has only {len(df)} rows, minimum required is {min_rows}")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_messages.append(f"Missing required columns: {missing_columns}")
    
    for column in df.columns:
        if df[column].isnull().any():
            null_count = df[column].isnull().sum()
            validation_messages.append(f"Column '{column}' has {null_count} missing values")
    
    if df.duplicated().any():
        dup_count = df.duplicated().sum()
        validation_messages.append(f"Dataset contains {dup_count} duplicate rows")
    
    is_valid = len(validation_messages) == 0
    validation_message = "Dataset is valid" if is_valid else "; ".join(validation_messages)
    
    return is_valid, validation_message

def get_dataset_summary(df):
    """
    Generate a summary of the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank', 'Frank'],
        'age': [25, 30, 30, 35, None, 40, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    summary = get_dataset_summary(df)
    print("Dataset summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    is_valid, message = validate_dataset(df, required_columns=['id', 'name', 'age', 'score'])
    print(f"Validation: {message}")
    
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)