import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Load a CSV file, handle missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values.")
            
            if missing_strategy == 'mean':
                # Fill numeric columns with mean
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                # Fill non-numeric with mode
                non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
                for col in non_numeric_cols:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            elif missing_strategy == 'drop':
                df = df.dropna()
            else:
                raise ValueError("Strategy must be 'mean' or 'drop'")
        
        # Remove duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate rows.")
            df = df.drop_duplicates()
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}. New shape: {df.shape}")
        return True
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    clean_csv_data('raw_data.csv', 'cleaned_data.csv', missing_strategy='mean')import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Cleans a pandas DataFrame by removing duplicates and standardizing column names.
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()

    # Standardize column names: lowercase and replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')

    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    removed_rows = initial_rows - df_clean.shape[0]

    # Fill missing numeric values with column median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Fill missing categorical values with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')

    # Remove leading/trailing whitespace from string columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].str.strip()

    # Log cleaning summary
    print(f"Cleaning complete. Removed {removed_rows} duplicate rows.")
    print(f"Final dataset shape: {df_clean.shape}")
    
    return df_clean

def validate_data(df):
    """
    Performs basic validation on the cleaned DataFrame.
    """
    validation_results = {
        'has_duplicates': df.duplicated().any(),
        'missing_values': df.isnull().sum().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Customer ID': [1, 2, 2, 3, 4],
        'First Name': ['John', 'Jane', 'Jane', 'Bob', None],
        'Last Name': ['Doe', 'Smith', 'Smith', 'Johnson', 'Williams'],
        'Age': [25, 30, 30, None, 35],
        'Purchase Amount': [100.50, 200.75, 200.75, 150.00, 300.25]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    validation = validate_data(cleaned_df)
    print("\nValidation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling to range [0, 1].
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    result = data.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if strategy == 'drop':
            result = result.dropna(subset=[col])
        elif strategy == 'mean':
            result[col] = result[col].fillna(result[col].mean())
        elif strategy == 'median':
            result[col] = result[col].fillna(result[col].median())
        elif strategy == 'mode':
            result[col] = result[col].fillna(result[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return result

def create_clean_data_pipeline(data, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Args:
        data: pandas DataFrame
        config: dictionary with cleaning operations configuration
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if 'remove_outliers' in config:
        for col, params in config['remove_outliers'].items():
            factor = params.get('factor', 1.5)
            cleaned_data = remove_outliers_iqr(cleaned_data, col, factor)
    
    if 'normalize' in config:
        for col in config['normalize']:
            if col in cleaned_data.columns:
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
    
    if 'standardize' in config:
        for col in config['standardize']:
            if col in cleaned_data.columns:
                cleaned_data[col] = standardize_zscore(cleaned_data, col)
    
    if 'handle_missing' in config:
        missing_config = config['handle_missing']
        strategy = missing_config.get('strategy', 'mean')
        columns = missing_config.get('columns', None)
        cleaned_data = handle_missing_values(cleaned_data, strategy, columns)
    
    return cleaned_data