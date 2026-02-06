import csv
import re
from datetime import datetime
from typing import List, Dict, Optional

def validate_email(email: str) -> bool:
    """Validate email format using regex pattern."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def clean_phone_number(phone: str) -> Optional[str]:
    """Remove non-digit characters from phone number."""
    digits = re.sub(r'\D', '', phone)
    return digits if len(digits) >= 10 else None

def parse_date(date_str: str, fmt: str = '%Y-%m-%d') -> Optional[datetime]:
    """Parse date string to datetime object."""
    try:
        return datetime.strptime(date_str.strip(), fmt)
    except ValueError:
        return None

def process_csv_row(row: Dict[str, str]) -> Dict[str, str]:
    """Process and clean individual CSV row data."""
    processed = row.copy()
    
    # Clean phone number
    if 'phone' in processed:
        processed['phone'] = clean_phone_number(processed['phone'])
    
    # Validate email
    if 'email' in processed:
        if not validate_email(processed['email']):
            processed['email'] = ''
    
    # Format date
    if 'date' in processed:
        date_obj = parse_date(processed['date'])
        processed['date'] = date_obj.strftime('%Y-%m-%d') if date_obj else ''
    
    return processed

def read_csv_file(filepath: str) -> List[Dict[str, str]]:
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

def write_csv_file(filepath: str, data: List[Dict[str, str]], fieldnames: List[str]) -> bool:
    """Write processed data to CSV file."""
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return False

def filter_valid_records(data: List[Dict[str, str]], required_fields: List[str]) -> List[Dict[str, str]]:
    """Filter records that have all required fields populated."""
    valid_records = []
    for record in data:
        if all(record.get(field) for field in required_fields):
            valid_records.append(record)
    return valid_records

def main():
    """Main function to demonstrate CSV processing."""
    input_file = 'input_data.csv'
    output_file = 'processed_data.csv'
    
    # Read raw data
    raw_data = read_csv_file(input_file)
    if not raw_data:
        return
    
    # Process each record
    processed_data = []
    for record in raw_data:
        processed_record = process_csv_row(record)
        processed_data.append(processed_record)
    
    # Filter valid records
    required_fields = ['name', 'email', 'phone']
    valid_data = filter_valid_records(processed_data, required_fields)
    
    # Write processed data
    if valid_data:
        fieldnames = list(valid_data[0].keys())
        success = write_csv_file(output_file, valid_data, fieldnames)
        if success:
            print(f"Successfully processed {len(valid_data)} records.")
            print(f"Output saved to: {output_file}")

if __name__ == '__main__':
    main()