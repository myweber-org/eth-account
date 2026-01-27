
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    numeric_columns (list): List of column names to clean.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[10, 'value'] = 500  # Add an outlier
    
    print("Original dataset shape:", df.shape)
    print("Original statistics:", calculate_summary_statistics(df, 'value'))
    
    cleaned_df = clean_dataset(df, ['value'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'value'))
import pandas as pd
import numpy as np
import re

def clean_csv_data(input_file, output_file):
    """
    Clean and preprocess CSV data by handling missing values,
    standardizing formats, and removing duplicates.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Standardize column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('unknown')
            else:
                df[col] = df[col].fillna(df[col].median())
        
        # Remove special characters from text columns
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaning complete. Cleaned file saved as: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

def validate_data(file_path):
    """
    Validate the cleaned data for basic quality checks.
    """
    try:
        df = pd.read_csv(file_path)
        
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        print("Data Validation Results:")
        for key, value in validation_results.items():
            print(f"{key}: {value}")
        
        return validation_results
        
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    if clean_csv_data(input_csv, output_csv):
        validate_data(output_csv)import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    """
    Load a CSV file and perform basic data cleaning.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Remove duplicate rows
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - df.shape[0]
    print(f"Removed {duplicates_removed} duplicate rows.")

    # Handle missing values: drop rows where all values are NaN
    df = df.dropna(how='all')
    # For numeric columns, fill missing values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.median()))

    # Remove outliers using IQR method for numeric columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter out outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print(f"Data shape after cleaning: {df.shape}")
    return df

def normalize_numeric_columns(df):
    """
    Normalize numeric columns to range [0, 1] using min-max scaling.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:  # Avoid division by zero
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0
    print("Numeric columns normalized.")
    return df

def save_cleaned_data(df, output_filepath):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    try:
        df.to_csv(output_filepath, index=False)
        print(f"Cleaned data saved to {output_filepath}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"

    cleaned_df = load_and_clean_data(input_file)
    if cleaned_df is not None:
        normalized_df = normalize_numeric_columns(cleaned_df)
        save_cleaned_data(normalized_df, output_file)