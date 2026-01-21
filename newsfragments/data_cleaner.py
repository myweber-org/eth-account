
import csv
import re
from typing import List, Dict, Optional

def read_csv_file(file_path: str) -> List[Dict]:
    """Read a CSV file and return a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    return data

def clean_string(value: str) -> str:
    """Remove extra whitespace and normalize string."""
    if not isinstance(value, str):
        return value
    cleaned = re.sub(r'\s+', ' ', value.strip())
    return cleaned

def clean_numeric(value: str) -> Optional[float]:
    """Convert string to float, handling common issues."""
    if not value:
        return None
    cleaned = value.replace(',', '').replace('$', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return None

def validate_email(email: str) -> bool:
    """Basic email validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def clean_csv_data(data: List[Dict]) -> List[Dict]:
    """Apply cleaning functions to all rows in the dataset."""
    cleaned_data = []
    for row in data:
        cleaned_row = {}
        for key, value in row.items():
            if key.lower().endswith('email'):
                cleaned_row[key] = value if validate_email(value) else ''
            elif any(num_key in key.lower() for num_key in ['price', 'amount', 'quantity']):
                cleaned_row[key] = clean_numeric(value)
            else:
                cleaned_row[key] = clean_string(value)
        cleaned_data.append(cleaned_row)
    return cleaned_data

def write_csv_file(data: List[Dict], file_path: str) -> bool:
    """Write cleaned data to a new CSV file."""
    if not data:
        return False
    try:
        fieldnames = data[0].keys()
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return False

def process_csv(input_path: str, output_path: str) -> None:
    """Complete CSV processing pipeline."""
    print(f"Reading data from {input_path}")
    raw_data = read_csv_file(input_path)
    
    if not raw_data:
        print("No data to process.")
        return
    
    print(f"Cleaning {len(raw_data)} rows...")
    cleaned_data = clean_csv_data(raw_data)
    
    print(f"Writing cleaned data to {output_path}")
    success = write_csv_file(cleaned_data, output_path)
    
    if success:
        print("CSV processing completed successfully.")
    else:
        print("CSV processing failed.")