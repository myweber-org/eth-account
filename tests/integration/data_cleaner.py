import pandas as pd
import numpy as np

def load_data(filepath):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method for specified column."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'")
    
    return filtered_df

def normalize_column(df, column):
    """Normalize column values to range [0, 1]."""
    if column not in df.columns:
        print(f"Error: Column '{column}' not found in dataframe.")
        return df
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        print(f"Warning: Column '{column}' has constant values. Normalization skipped.")
        return df
    
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    print(f"Column '{column}' normalized successfully.")
    
    return df

def clean_dataset(df, numeric_columns):
    """Main cleaning pipeline for numeric columns."""
    if df is None or df.empty:
        print("Error: Invalid dataframe provided.")
        return df
    
    original_shape = df.shape
    
    for column in numeric_columns:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
            df = normalize_column(df, column)
        else:
            print(f"Warning: Column '{column}' not found. Skipping.")
    
    print(f"Data cleaning complete. Original shape: {original_shape}, Cleaned shape: {df.shape}")
    return df

def save_cleaned_data(df, output_path):
    """Save cleaned dataframe to CSV."""
    if df is not None and not df.empty:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return True
    else:
        print("Error: No data to save.")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    numeric_cols = ['age', 'income', 'score']
    
    raw_data = load_data(input_file)
    
    if raw_data is not None:
        cleaned_data = clean_dataset(raw_data, numeric_cols)
        save_cleaned_data(cleaned_data, output_file)
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize column using min-max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize column using z-score normalization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_multiplier=1.5, normalize_method='minmax'):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove outliers
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_multiplier)
            
            # Apply normalization
            if normalize_method == 'minmax':
                cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
            elif normalize_method == 'zscore':
                cleaned_df[f'{col}_standardized'] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df, required_columns, min_rows=1):
    """
    Validate dataframe structure and content.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    return True

def get_summary_statistics(df, numeric_columns):
    """
    Generate summary statistics for numeric columns.
    """
    summary = {}
    
    for col in numeric_columns:
        if col in df.columns:
            summary[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': df[col].count(),
                'missing': df[col].isnull().sum()
            }
    
    return pd.DataFrame(summary).T

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'id': range(100),
        'feature_a': np.random.normal(50, 15, 100),
        'feature_b': np.random.exponential(2, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data shape:", df.shape)
    
    # Clean the data
    numeric_cols = ['feature_a', 'feature_b']
    cleaned_df = clean_dataset(df, numeric_cols, normalize_method='minmax')
    print("Cleaned data shape:", cleaned_df.shape)
    
    # Get summary statistics
    summary = get_summary_statistics(cleaned_df, numeric_cols)
    print("\nSummary statistics:")
    print(summary)