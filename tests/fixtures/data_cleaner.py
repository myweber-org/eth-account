
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        subset (list, optional): Column names to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df.reset_index(drop=True)

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    df_copy = dataframe.copy()
    
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    return df_copy

def validate_dataframe(dataframe, required_columns):
    """
    Validate that DataFrame contains required columns and is not empty.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if dataframe.empty:
        print("Error: DataFrame is empty")
        return False
    
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return False
    
    return True

def save_cleaned_data(dataframe, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to save
        output_path (str): Path to save the file
        format (str): File format ('csv', 'parquet', 'json')
    
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        if format == 'csv':
            dataframe.to_csv(output_path, index=False)
        elif format == 'parquet':
            dataframe.to_parquet(output_path, index=False)
        elif format == 'json':
            dataframe.to_json(output_path, orient='records')
        else:
            print(f"Error: Unsupported format '{format}'")
            return False
        
        print(f"Data saved successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False