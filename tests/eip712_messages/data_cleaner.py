import csv
import hashlib
from collections import defaultdict

def generate_row_hash(row):
    """Generate a hash for a CSV row to identify duplicates."""
    row_string = ''.join(str(field) for field in row)
    return hashlib.md5(row_string.encode()).hexdigest()

def remove_duplicates(input_file, output_file, key_columns=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file: Path to the input CSV file.
        output_file: Path to the output CSV file.
        key_columns: List of column indices to consider for duplicate detection.
                     If None, consider all columns.
    """
    seen_hashes = set()
    unique_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers = next(reader)
        
        for row in reader:
            if key_columns is not None:
                # Consider only specified columns for duplicate detection
                key_data = [row[i] for i in key_columns]
                row_hash = generate_row_hash(key_data)
            else:
                # Consider all columns for duplicate detection
                row_hash = generate_row_hash(row)
            
            if row_hash not in seen_hashes:
                seen_hashes.add(row_hash)
                unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
        writer.writerows(unique_rows)
    
    print(f"Removed {len(seen_hashes) - len(unique_rows)} duplicate rows.")
    print(f"Original rows: {len(seen_hashes)}")
    print(f"Unique rows: {len(unique_rows)}")

def find_duplicate_counts(input_file, key_columns=None):
    """
    Analyze duplicate frequency in a CSV file.
    
    Args:
        input_file: Path to the input CSV file.
        key_columns: List of column indices to consider for duplicate detection.
    
    Returns:
        Dictionary with duplicate hashes and their counts.
    """
    hash_counter = defaultdict(int)
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers = next(reader)
        
        for row in reader:
            if key_columns is not None:
                key_data = [row[i] for i in key_columns]
                row_hash = generate_row_hash(key_data)
            else:
                row_hash = generate_row_hash(row)
            
            hash_counter[row_hash] += 1
    
    # Filter to only show duplicates
    duplicates = {h: c for h, c in hash_counter.items() if c > 1}
    
    return duplicates

if __name__ == "__main__":
    # Example usage
    input_csv = "data.csv"
    output_csv = "cleaned_data.csv"
    
    # Remove duplicates considering all columns
    remove_duplicates(input_csv, output_csv)
    
    # Analyze duplicates based on specific columns (e.g., first two columns)
    duplicate_stats = find_duplicate_counts(input_csv, key_columns=[0, 1])
    print(f"Found {len(duplicate_stats)} duplicate groups based on columns 0 and 1")