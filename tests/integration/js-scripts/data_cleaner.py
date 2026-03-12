
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list (list): The list from which duplicates should be removed.
    
    Returns:
        list: A new list with duplicates removed.
    """
    seen = set()
    result = []
    
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_numeric_data(values, default=0):
    """
    Clean a list of numeric values by converting non-numeric entries to default.
    
    Args:
        values (list): List of values to clean.
        default (int/float): Default value for non-numeric entries.
    
    Returns:
        list: Cleaned list of numeric values.
    """
    cleaned = []
    
    for value in values:
        try:
            cleaned.append(float(value))
        except (ValueError, TypeError):
            cleaned.append(default)
    
    return cleaned

def filter_by_threshold(data, threshold, key=None):
    """
    Filter data based on a threshold value.
    
    Args:
        data (list): Data to filter.
        threshold: Minimum value to include.
        key (callable): Function to extract value from each element.
    
    Returns:
        list: Filtered data.
    """
    if key is None:
        key = lambda x: x
    
    return [item for item in data if key(item) >= threshold]

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    
    mixed_data = [1, "2", 3.5, "invalid", 4]
    print("Numeric cleaned:", clean_numeric_data(mixed_data))
    
    values = [10, 5, 20, 3, 15]
    print("Filtered (>=10):", filter_by_threshold(values, 10))
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_strategy='mean', output_path=None):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Parameters:
    file_path (str): Path to input CSV file.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero').
    output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
    
    Returns:
    pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            print("Missing values per column:")
            print(missing_counts[missing_counts > 0])
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            
            for col in df.columns:
                if df[col].isnull().any():
                    if col in numeric_cols:
                        if fill_strategy == 'mean':
                            fill_value = df[col].mean()
                        elif fill_strategy == 'median':
                            fill_value = df[col].median()
                        elif fill_strategy == 'mode':
                            fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
                        elif fill_strategy == 'zero':
                            fill_value = 0
                        else:
                            fill_value = df[col].mean()
                        df[col].fillna(fill_value, inplace=True)
                        print(f"Filled missing values in '{col}' with {fill_strategy}: {fill_value:.2f}")
                    elif col in categorical_cols:
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                        df[col].fillna(mode_val, inplace=True)
                        print(f"Filled missing values in '{col}' with mode: '{mode_val}'")
        else:
            print("No missing values found.")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def summarize_data(df):
    """
    Generate summary statistics for a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    dict: Summary statistics.
    """
    if df is None or df.empty:
        return {}
    
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'numeric_summary': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {},
        'categorical_summary': {col: df[col].value_counts().to_dict() 
                               for col in df.select_dtypes(exclude=[np.number]).columns}
    }
    return summary

if __name__ == "__main__":
    # Example usage
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='median')
    if cleaned_df is not None:
        summary = summarize_data(cleaned_df)
        print(f"Data shape: {summary.get('shape')}")
        print(f"Columns: {summary.get('columns')}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
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
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has less than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 10 + 50,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original DataFrame shape: {df.shape}")
    
    # Add some outliers
    df.loc[0, 'value'] = 200
    df.loc[1, 'value'] = -100
    
    # Clean the data
    cleaned_df = clean_numeric_data(df, columns=['value'])
    print(f"Cleaned DataFrame shape: {cleaned_df.shape}")
    
    # Validate the cleaned data
    is_valid = validate_data(cleaned_df, required_columns=['id', 'value', 'category'], min_rows=10)
    print(f"Data validation passed: {is_valid}")