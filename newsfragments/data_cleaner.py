
import pandas as pd

def clean_dataset(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_na (bool): Whether to drop rows with null values
    rename_columns (bool): Whether to standardize column names
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df.columns = (
            cleaned_df.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace('[^a-z0-9_]', '', regex=True)
        )
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    if required_columns:
        missing_columns = [
            col for col in required_columns 
            if col not in df.columns
        ]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(
                f"Missing required columns: {missing_columns}"
            )
    
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['errors'].append("DataFrame is empty")
    
    return validation_result

if __name__ == "__main__":
    sample_data = {
        'First Name': ['John', 'Jane', None, 'Bob'],
        'Last Name': ['Doe', 'Smith', 'Johnson', None],
        'Age': [25, 30, 35, 40],
        'Email Address': ['john@example.com', 'jane@test.com', None, 'bob@sample.com']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df)
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['first_name', 'last_name'])
    print("\nValidation Result:")
    print(validation)
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
        subset (list): Columns to consider for duplicates (optional)
        keep (str): Which duplicate to keep - 'first', 'last', or False to drop all
    """
    try:
        df = pd.read_csv(input_file)
        
        if subset:
            df_clean = df.drop_duplicates(subset=subset, keep=keep)
        else:
            df_clean = df.drop_duplicates(keep=keep)
        
        if output_file:
            df_clean.to_csv(output_file, index=False)
            print(f"Cleaned data saved to {output_file}")
            print(f"Removed {len(df) - len(df_clean)} duplicate rows")
        else:
            df_clean.to_csv(input_file, index=False)
            print(f"Original file updated: {input_file}")
            print(f"Removed {len(df) - len(df_clean)} duplicate rows")
            
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)