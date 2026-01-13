
import pandas as pd
import numpy as np

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by standardizing columns, removing duplicates,
    and handling missing values.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def standardize_strings(df, columns):
    """
    Standardize string columns by converting to lowercase and stripping whitespace.
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.lower().str.strip()
    return df_copy

def validate_data(df, required_columns, date_columns=None):
    """
    Validate DataFrame structure and data types.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if date_columns:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

def process_data_file(input_path, output_path, config):
    """
    Main function to process a data file with cleaning operations.
    """
    try:
        df = pd.read_csv(input_path)
        
        df = clean_dataframe(
            df,
            column_mapping=config.get('column_mapping'),
            drop_duplicates=config.get('drop_duplicates', True),
            fill_missing=config.get('fill_missing', True)
        )
        
        if config.get('standardize_columns'):
            df = standardize_strings(df, config['standardize_columns'])
        
        if config.get('required_columns'):
            df = validate_data(
                df,
                config['required_columns'],
                config.get('date_columns')
            )
        
        df.to_csv(output_path, index=False)
        print(f"Data processed successfully. Output saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return False

if __name__ == "__main__":
    config = {
        'column_mapping': {'old_name': 'new_name'},
        'drop_duplicates': True,
        'fill_missing': True,
        'standardize_columns': ['name', 'category'],
        'required_columns': ['id', 'name', 'value'],
        'date_columns': ['date']
    }
    
    process_data_file('input_data.csv', 'cleaned_data.csv', config)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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
    
    outliers_removed = len(df) - len(filtered_df)
    if outliers_removed > 0:
        print(f"Removed {outliers_removed} outliers from column '{column}'")
    
    return filtered_df.reset_index(drop=True)

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame containing only the outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return outliers

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
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Error processing column '{col}': {e}")
    
    return cleaned_df