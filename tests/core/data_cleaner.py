
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean a CSV file by handling missing values.
    
    Parameters:
    file_path (str): Path to the CSV file.
    fill_strategy (str): Strategy for filling missing values.
                         Options: 'mean', 'median', 'mode', 'zero'.
    drop_threshold (float): Drop columns with missing ratio above this threshold.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    df = pd.read_csv(file_path)
    
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if fill_strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif fill_strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif fill_strategy == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    for col in categorical_cols:
        mode_value = df[col].mode()
        if not mode_value.empty:
            df[col] = df[col].fillna(mode_value.iloc[0])
        else:
            df[col] = df[col].fillna('Unknown')
    
    df = df.reset_index(drop=True)
    
    return df

def export_cleaned_data(df, output_path):
    """
    Export cleaned DataFrame to CSV.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame.
    output_path (str): Path for output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data exported to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, fill_strategy='mean', drop_threshold=0.5)
    export_cleaned_data(cleaned_df, output_file)