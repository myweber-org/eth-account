
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(100),
        'value': np.random.randn(100) * 10 + 50,
        'score': np.random.randn(100) * 5 + 75
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("\nSummary statistics:")
    print(df[['value', 'score']].describe())
    
    cleaned_df = clean_numeric_data(df, ['value', 'score'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned summary statistics:")
    print(cleaned_df[['value', 'score']].describe())
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: Dictionary mapping old column names to new ones
        drop_duplicates: Whether to remove duplicate rows
        normalize_text: Whether to normalize text columns (strip, lower case)
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    return df_clean

def validate_email(email):
    """
    Validate email format using regex.
    
    Args:
        email: String containing email address
    
    Returns:
        Boolean indicating if email is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def remove_special_characters(text):
    """
    Remove special characters from text, keeping only alphanumeric and spaces.
    
    Args:
        text: String to clean
    
    Returns:
        Cleaned string
    """
    if pd.isna(text):
        return text
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text))

def main():
    sample_data = {
        'Name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson  '],
        'Email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.net'],
        'Age': [25, 30, 25, 35],
        'Notes': ['Important!', 'Needs follow-up', 'Urgent!!!', 'Regular']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataframe(
        df, 
        column_mapping={'Notes': 'Comments'},
        drop_duplicates=True,
        normalize_text=True
    )
    
    cleaned_df['Email_Valid'] = cleaned_df['Email'].apply(validate_email)
    cleaned_df['Name_Clean'] = cleaned_df['Name'].apply(remove_special_characters)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = main()
import pandas as pd
import sys

def remove_duplicates(input_file, output_file, subset=None):
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        if subset:
            df_cleaned = df.drop_duplicates(subset=subset)
        else:
            df_cleaned = df.drop_duplicates()
        final_count = len(df_cleaned)
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaning complete. Removed {initial_count - final_count} duplicate rows.")
        print(f"Original rows: {initial_count}, Cleaned rows: {final_count}")
        print(f"Cleaned data saved to: {output_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python data_cleaner.py <input_file.csv> <output_file.csv> [subset_columns]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")
        print("To check duplicates based on specific columns, provide them as a comma-separated string.")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    subset = None
    if len(sys.argv) > 3:
        subset = [col.strip() for col in sys.argv[3].split(',')]
    remove_duplicates(input_file, output_file, subset)
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if fill_missing == 'drop':
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna()
        print(f"Removed {initial_rows - len(cleaned_df)} rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"Filled missing values in column '{column}' with {fill_missing}: {fill_value}")
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate the structure and content of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    numeric_columns (list): List of column names that should be numeric.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'non_numeric_columns': [],
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['missing_columns'] = missing_cols
            validation_results['is_valid'] = False
    
    if numeric_columns:
        non_numeric = [col for col in numeric_columns if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            validation_results['non_numeric_columns'] = non_numeric
            validation_results['is_valid'] = False
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None],
        'B': [10, 20, 20, None, 50, 60],
        'C': ['x', 'y', 'y', 'z', 'x', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_data(cleaned, required_columns=['A', 'B', 'C'], numeric_columns=['A', 'B'])
    print("\nValidation Results:")
    print(validation)