
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