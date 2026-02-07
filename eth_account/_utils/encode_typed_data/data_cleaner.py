import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_numeric_data(df, numeric_columns):
    """
    Clean numeric columns by removing outliers and filling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    return cleaned_df

def validate_dataframe(df, required_columns):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

def process_data_file(file_path, numeric_cols, required_cols):
    """
    Main function to process a data file.
    
    Parameters:
    file_path (str): Path to the data file
    numeric_cols (list): List of numeric column names
    required_cols (list): List of required column names
    
    Returns:
    pd.DataFrame: Processed DataFrame or None if error occurs
    """
    try:
        df = pd.read_csv(file_path)
        
        if not validate_dataframe(df, required_cols):
            return None
        
        processed_df = clean_numeric_data(df, numeric_cols)
        
        print(f"Original shape: {df.shape}")
        print(f"Cleaned shape: {processed_df.shape}")
        print(f"Rows removed: {len(df) - len(processed_df)}")
        
        return processed_df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[10:15, 'value'] = np.nan
    df.loc[95:99, 'value'] = 500
    
    numeric_columns = ['value']
    required_columns = ['id', 'value', 'category']
    
    result = process_data_file('sample_data.csv', numeric_columns, required_columns)
    
    if result is not None:
        print("Data cleaning completed successfully")
        print(result.head())