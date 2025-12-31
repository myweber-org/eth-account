import csv
import re

def clean_csv(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        cleaned_rows = []
        for row in reader:
            cleaned_row = {}
            for key, value in row.items():
                if value is None:
                    cleaned_row[key] = ''
                else:
                    cleaned_value = re.sub(r'\s+', ' ', value.strip())
                    cleaned_value = cleaned_value.replace('\n', ' ')
                    cleaned_row[key] = cleaned_value
            cleaned_rows.append(cleaned_row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

def remove_duplicates(input_file, output_file, key_column):
    seen = set()
    unique_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            key_value = row.get(key_column, '')
            if key_value not in seen:
                seen.add(key_value)
                unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)

def validate_email_column(input_file, email_column):
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    invalid_emails = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        
        for i, row in enumerate(reader, start=2):
            email = row.get(email_column, '')
            if email and not re.match(email_pattern, email):
                invalid_emails.append({
                    'row': i,
                    'email': email
                })
    
    return invalid_emails
import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = cleaned_df.shape[0]
    cleaned_df.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - cleaned_df.shape[0]
    
    # Handle missing values
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns
    
    missing_counts = {}
    for column in columns_to_check:
        if column in cleaned_df.columns:
            missing_count = cleaned_df[column].isnull().sum()
            if missing_count > 0:
                # For numeric columns, fill with median
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    median_value = cleaned_df[column].median()
                    cleaned_df[column].fillna(median_value, inplace=True)
                # For categorical columns, fill with mode
                else:
                    mode_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown'
                    cleaned_df[column].fillna(mode_value, inplace=True)
                missing_counts[column] = missing_count
    
    # Print cleaning summary
    print(f"Removed {removed_duplicates} duplicate rows")
    print(f"Handled missing values in {len(missing_counts)} columns:")
    for column, count in missing_counts.items():
        print(f"  - {column}: {count} missing values filled")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'name': ['Alice', 'Bob', 'Bob', None, 'Eve', None],
#         'age': [25, 30, 30, 35, None, 40],
#         'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaning data...")
#     
#     cleaned = clean_dataset(df)
#     print("\nCleaned DataFrame:")
#     print(cleaned)