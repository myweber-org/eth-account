import csv
import re
from typing import List, Dict, Optional

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def clean_phone_number(phone: str) -> Optional[str]:
    """Remove non-digit characters from phone number."""
    digits = re.sub(r'\D', '', phone)
    return digits if len(digits) >= 10 else None

def read_csv_file(filepath: str) -> List[Dict]:
    """Read CSV file and return list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return data

def clean_data(rows: List[Dict]) -> List[Dict]:
    """Clean and validate data rows."""
    cleaned_rows = []
    for row in rows:
        cleaned_row = row.copy()
        
        # Clean phone number
        if 'phone' in cleaned_row:
            cleaned_row['phone'] = clean_phone_number(cleaned_row['phone'])
        
        # Validate email
        if 'email' in cleaned_row:
            cleaned_row['email_valid'] = validate_email(cleaned_row['email'])
        
        # Trim string fields
        for key, value in cleaned_row.items():
            if isinstance(value, str):
                cleaned_row[key] = value.strip()
        
        cleaned_rows.append(cleaned_row)
    
    return cleaned_rows

def write_csv_file(filepath: str, data: List[Dict], fieldnames: List[str]):
    """Write data to CSV file."""
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Successfully wrote cleaned data to '{filepath}'")
    except Exception as e:
        print(f"Error writing CSV: {e}")

def process_csv(input_file: str, output_file: str):
    """Main function to process CSV file."""
    raw_data = read_csv_file(input_file)
    if not raw_data:
        return
    
    cleaned_data = clean_data(raw_data)
    
    # Get fieldnames from first row
    if cleaned_data:
        fieldnames = list(cleaned_data[0].keys())
        write_csv_file(output_file, cleaned_data, fieldnames)

if __name__ == "__main__":
    process_csv('input_data.csv', 'cleaned_data.csv')