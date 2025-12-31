
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    removed_duplicates = initial_rows - df_clean.shape[0]
    
    # Normalize specified string columns
    if columns_to_clean is None:
        # Automatically detect object/string columns
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    def normalize_string(text):
        if pd.isna(text):
            return text
        # Convert to string, strip whitespace, and lowercase
        text = str(text).strip().lower()
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text
    
    for col in columns_to_clean:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(normalize_string)
    
    return df_clean, removed_duplicates

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column and add a validation flag.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    df_validated = df.copy()
    df_validated['email_valid'] = df_validated[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    valid_count = df_validated['email_valid'].sum()
    invalid_count = len(df_validated) - valid_count
    
    return df_validated, valid_count, invalid_count

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file in specified format.
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'excel', or 'json'.")
    
    return output_path