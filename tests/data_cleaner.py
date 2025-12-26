import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    filepath (str): Path to the CSV file
    fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Drop columns with missing ratio above this threshold
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    # Calculate missing percentage per column
    missing_percent = df.isnull().sum() / len(df) * 100
    
    # Drop columns with too many missing values
    columns_to_drop = missing_percent[missing_percent > drop_threshold * 100].index
    if len(columns_to_drop) > 0:
        print(f"Dropping columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
        df = df.drop(columns=columns_to_drop)
    
    # Fill remaining missing values based on specified method
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_method == 'mean':
        fill_values = df[numeric_cols].mean()
    elif fill_method == 'median':
        fill_values = df[numeric_cols].median()
    elif fill_method == 'zero':
        fill_values = 0
    elif fill_method == 'mode':
        fill_values = df[numeric_cols].mode().iloc[0]
    else:
        print(f"Warning: Unknown fill method '{fill_method}', using mean instead")
        fill_values = df[numeric_cols].mean()
    
    # Fill numeric columns
    df[numeric_cols] = df[numeric_cols].fillna(fill_values)
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_value)
    
    # Report cleaning results
    final_missing = df.isnull().sum().sum()
    if final_missing > 0:
        print(f"Warning: {final_missing} missing values remain after cleaning")
    else:
        print("All missing values have been handled")
    
    print(f"Final data shape: {df.shape}")
    print(f"Removed {original_shape[1] - df.shape[1]} columns")
    
    return df

def export_cleaned_data(df, output_path):
    """
    Export cleaned dataframe to CSV.
    
    Parameters:
    df (pd.DataFrame): Cleaned dataframe
    output_path (str): Path for output CSV file
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data exported to {output_path}")
        return True
    return False

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, fill_method='median', drop_threshold=0.3)
    
    if cleaned_df is not None:
        export_cleaned_data(cleaned_df, output_file)