
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to numeric and filling NaN with mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
    return df

def validate_dataframe(df, required_columns):
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if all required columns are present.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    return Trueimport pandas as pd
import numpy as np
import logging

def clean_csv_data(input_file, output_file, drop_na=True, fill_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_file (str): Path to input CSV file.
        output_file (str): Path to save cleaned CSV file.
        drop_na (bool): If True, drop rows with missing values. If False, fill them.
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode').
    
    Returns:
        bool: True if cleaning successful, False otherwise.
    """
    try:
        df = pd.read_csv(input_file)
        logging.info(f"Loaded data from {input_file} with shape {df.shape}")
        
        original_rows = df.shape[0]
        df = df.drop_duplicates()
        logging.info(f"Removed {original_rows - df.shape[0]} duplicate rows")
        
        if drop_na:
            df = df.dropna()
            logging.info(f"Dropped rows with missing values. New shape: {df.shape}")
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if fill_strategy == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_strategy == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_strategy == 'mode':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
            logging.info(f"Filled missing values using {fill_strategy} strategy")
        
        df.to_csv(output_file, index=False)
        logging.info(f"Saved cleaned data to {output_file}")
        return True
        
    except FileNotFoundError:
        logging.error(f"Input file {input_file} not found")
        return False
    except pd.errors.EmptyDataError:
        logging.error(f"Input file {input_file} is empty")
        return False
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return False

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pd.DataFrame): Dataframe to validate.
        required_columns (list): List of required column names.
    
    Returns:
        dict: Validation results with keys 'is_valid' and 'message'.
    """
    validation_result = {'is_valid': True, 'message': 'Validation passed'}
    
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['message'] = 'Dataframe is empty'
        return validation_result
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['message'] = f'Missing required columns: {missing_columns}'
            return validation_result
    
    return validation_result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, None, 20.1, 20.1],
        'category': ['A', 'B', 'A', 'B', 'A', 'A']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_input.csv', index=False)
    
    success = clean_csv_data('test_input.csv', 'test_output.csv', drop_na=False, fill_strategy='mean')
    
    if success:
        cleaned_df = pd.read_csv('test_output.csv')
        print(f"Cleaned dataframe shape: {cleaned_df.shape}")
        print(cleaned_df.head())
    
    import os
    if os.path.exists('test_input.csv'):
        os.remove('test_input.csv')
    if os.path.exists('test_output.csv'):
        os.remove('test_output.csv')