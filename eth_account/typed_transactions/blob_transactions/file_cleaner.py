
import os
import shutil
import argparse

def clean_directory(directory, extensions_to_remove):
    """
    Remove files with specified extensions from the given directory.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    removed_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions_to_remove):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    removed_files.append(file_path)
                    print(f"Removed: {file_path}")
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")

    return removed_files

def main():
    parser = argparse.ArgumentParser(description="Clean temporary files from a directory.")
    parser.add_argument("directory", help="Directory to clean")
    parser.add_argument("-e", "--extensions", nargs="+", default=[".tmp", ".log", ".bak"],
                        help="File extensions to remove (default: .tmp .log .bak)")

    args = parser.parse_args()

    print(f"Cleaning directory: {args.directory}")
    print(f"Removing files with extensions: {args.extensions}")
    
    removed = clean_directory(args.directory, args.extensions)
    
    if removed:
        print(f"\nCleaning complete. Removed {len(removed)} file(s).")
    else:
        print("\nNo files matching the specified extensions were found.")

if __name__ == "__main__":
    main()
import os
import re
import unicodedata

def clean_filename(filename):
    """
    Normalize and clean a filename by removing or replacing invalid characters,
    converting to lowercase, and stripping extra spaces.
    """
    # Normalize unicode characters
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    
    # Replace spaces and invalid characters with underscores
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = re.sub(r'[\s]+', '_', filename)
    
    # Convert to lowercase and strip leading/trailing underscores/dots
    filename = filename.lower().strip('_.')
    
    return filename

def clean_filenames_in_directory(directory_path):
    """
    Iterate through files in a directory and rename them with cleaned filenames.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return
    
    for filename in os.listdir(directory_path):
        old_path = os.path.join(directory_path, filename)
        
        if os.path.isfile(old_path):
            new_filename = clean_filename(filename)
            new_path = os.path.join(directory_path, new_filename)
            
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                except OSError as e:
                    print(f"Error renaming {filename}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to clean filenames: ").strip()
    clean_filenames_in_directory(target_directory)
import os
import glob
import sys

def clean_temp_files(directory, patterns):
    """
    Remove files matching given patterns in the specified directory.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False
    
    removed_count = 0
    for pattern in patterns:
        search_path = os.path.join(directory, pattern)
        for file_path in glob.glob(search_path):
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
                removed_count += 1
            except OSError as e:
                print(f"Error removing {file_path}: {e}")
    
    print(f"Total files removed: {removed_count}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <directory>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    temp_patterns = ['*.tmp', '*.temp', '~*', '*.bak']
    
    clean_temp_files(target_dir, temp_patterns)