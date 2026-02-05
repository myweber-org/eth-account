
import sys
import os

def remove_duplicates(input_file, output_file=None):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return False
    
    if output_file is None:
        output_file = input_file + ".deduped"
    
    seen_lines = set()
    lines_removed = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in lines:
                stripped_line = line.rstrip('\n')
                if stripped_line not in seen_lines:
                    f.write(line)
                    seen_lines.add(stripped_line)
                else:
                    lines_removed += 1
        
        print(f"Successfully processed '{input_file}'")
        print(f"Unique lines kept: {len(seen_lines)}")
        print(f"Duplicate lines removed: {lines_removed}")
        print(f"Output saved to: '{output_file}'")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)
import os
import time
from pathlib import Path

def clean_old_files(directory, days=7):
    """
    Remove files in the specified directory that are older than the given number of days.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return
    
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    deleted_count = 0
    total_size = 0
    
    for item in Path(directory).rglob('*'):
        if item.is_file():
            try:
                file_mtime = item.stat().st_mtime
                if file_mtime < cutoff_time:
                    file_size = item.stat().st_size
                    item.unlink()
                    deleted_count += 1
                    total_size += file_size
                    print(f"Removed: {item}")
            except (OSError, PermissionError) as e:
                print(f"Failed to remove {item}: {e}")
    
    print(f"Cleaning completed. Removed {deleted_count} files, freed {total_size / (1024*1024):.2f} MB.")

if __name__ == "__main__":
    target_dir = "/tmp"
    clean_old_files(target_dir)
import os
import re
import sys

def normalize_filename(filename):
    """
    Normalize a filename by removing special characters,
    converting spaces to underscores, and making it lowercase.
    """
    # Remove any non-alphanumeric characters except dots, hyphens, and underscores
    normalized = re.sub(r'[^\w\.\-]', '_', filename)
    # Replace multiple underscores with a single one
    normalized = re.sub(r'_+', '_', normalized)
    # Convert to lowercase
    normalized = normalized.lower()
    # Remove leading/trailing underscores or dots
    normalized = normalized.strip('_.')
    return normalized

def clean_directory(directory_path):
    """
    Clean all filenames in the specified directory.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        sys.exit(1)

    for filename in os.listdir(directory_path):
        old_path = os.path.join(directory_path, filename)
        if os.path.isfile(old_path):
            new_name = normalize_filename(filename)
            new_path = os.path.join(directory_path, new_name)

            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_name}")
                except OSError as e:
                    print(f"Error renaming {filename}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_cleaner.py <directory_path>")
        sys.exit(1)

    target_directory = sys.argv[1]
    clean_directory(target_directory)