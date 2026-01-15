import csv
import sys
from pathlib import Path

def clean_csv(input_path, output_path=None):
    """
    Clean a CSV file by removing rows with missing values
    and stripping whitespace from all string fields.
    """
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = f"{input_stem}_cleaned.csv"
    
    cleaned_rows = []
    with open(input_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Skip rows with any empty values
            if any(value is None or str(value).strip() == '' for value in row.values()):
                continue
            
            # Strip whitespace from all string fields
            cleaned_row = {
                key: value.strip() if isinstance(value, str) else value
                for key, value in row.items()
            }
            cleaned_rows.append(cleaned_row)
    
    if cleaned_rows:
        with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_rows)
        
        print(f"Cleaned data saved to: {output_path}")
        print(f"Original rows: {len(cleaned_rows) + (len(cleaned_rows) - len(cleaned_rows))}")
        print(f"Cleaned rows: {len(cleaned_rows)}")
    else:
        print("No valid rows found after cleaning")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_csv> [output_csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        clean_csv(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
    except Exception as e:
        print(f"Error processing file: {e}")