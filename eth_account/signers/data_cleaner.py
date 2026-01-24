import csv
import re

def clean_string(value):
    if not isinstance(value, str):
        return str(value)
    value = value.strip()
    value = re.sub(r'\s+', ' ', value)
    return value

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def read_and_clean_csv(input_path, output_path):
    cleaned_rows = []
    with open(input_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            cleaned_row = {}
            for key, value in row.items():
                cleaned_row[key] = clean_string(value)
            cleaned_rows.append(cleaned_row)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    return len(cleaned_rows)

def filter_valid_emails(input_path, output_path):
    valid_rows = []
    with open(input_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            if 'email' in row and validate_email(row['email']):
                valid_rows.append(row)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_rows)
    
    return len(valid_rows)

if __name__ == "__main__":
    input_file = "raw_data.csv"
    cleaned_file = "cleaned_data.csv"
    filtered_file = "valid_emails.csv"
    
    count_cleaned = read_and_clean_csv(input_file, cleaned_file)
    print(f"Cleaned {count_cleaned} rows")
    
    count_valid = filter_valid_emails(cleaned_file, filtered_file)
    print(f"Found {count_valid} rows with valid emails")