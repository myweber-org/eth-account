import csv
import os
from typing import List, Dict, Any

def read_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """Read a CSV file and return a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    return data

def remove_empty_rows(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove rows where all values are empty strings."""
    cleaned_data = []
    for row in data:
        if any(value.strip() for value in row.values()):
            cleaned_data.append(row)
    return cleaned_data

def standardize_column_names(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Standardize column names to lowercase with underscores."""
    if not data:
        return data
    
    standardized_data = []
    for row in data:
        new_row = {}
        for key, value in row.items():
            new_key = key.strip().lower().replace(' ', '_')
            new_row[new_key] = value
        standardized_data.append(new_row)
    return standardized_data

def write_csv_file(data: List[Dict[str, Any]], file_path: str) -> bool:
    """Write data to a CSV file."""
    if not data:
        print("No data to write.")
        return False
    
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return False

def clean_csv_data(input_file: str, output_file: str) -> None:
    """Main function to clean CSV data."""
    print(f"Reading data from {input_file}")
    data = read_csv_file(input_file)
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    print(f"Original data: {len(data)} rows")
    
    data = remove_empty_rows(data)
    print(f"After removing empty rows: {len(data)} rows")
    
    data = standardize_column_names(data)
    print("Column names standardized")
    
    if write_csv_file(data, output_file):
        print(f"Cleaned data written to {output_file}")
    else:
        print("Failed to write cleaned data")
import csv
import os
from typing import List, Dict, Any

def read_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """Read a CSV file and return a list of dictionaries."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def clean_numeric_field(record: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    """Clean a numeric field by removing non-numeric characters and converting to float."""
    if field_name in record:
        value = record[field_name]
        if isinstance(value, str):
            cleaned_value = ''.join(ch for ch in value if ch.isdigit() or ch == '.')
            try:
                record[field_name] = float(cleaned_value) if cleaned_value else 0.0
            except ValueError:
                record[field_name] = 0.0
    return record

def remove_empty_rows(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove rows where all values are empty or whitespace."""
    cleaned_data = []
    for row in data:
        if any(value and str(value).strip() for value in row.values()):
            cleaned_data.append(row)
    return cleaned_data

def write_csv_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """Write data to a CSV file."""
    if not data:
        return
    
    fieldnames = data[0].keys()
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def process_csv(input_path: str, output_path: str, numeric_fields: List[str] = None) -> None:
    """Main function to clean and process a CSV file."""
    if numeric_fields is None:
        numeric_fields = []
    
    data = read_csv_file(input_path)
    data = remove_empty_rows(data)
    
    for record in data:
        for field in numeric_fields:
            record = clean_numeric_field(record, field)
    
    write_csv_file(data, output_path)
    print(f"Processed {len(data)} records. Output saved to {output_path}")

if __name__ == "__main__":
    sample_data = [
        {"name": "Alice", "age": "25", "score": "95.5"},
        {"name": "Bob", "age": "30", "score": "88.0"},
        {"name": "Charlie", "age": "abc", "score": "invalid"},
        {"name": "", "age": "", "score": ""}
    ]
    
    test_input = "test_input.csv"
    test_output = "test_output.csv"
    
    with open(test_input, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["name", "age", "score"])
        writer.writeheader()
        writer.writerows(sample_data)
    
    process_csv(test_input, test_output, numeric_fields=["age", "score"])
    
    os.remove(test_input)
    os.remove(test_output)