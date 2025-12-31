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