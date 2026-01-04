
import os
import sys
import argparse

def rename_files_with_sequential_numbering(directory, prefix="file", extension=".txt", start_number=1):
    """
    Rename all files in the specified directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files to rename
        prefix (str): Prefix for renamed files
        extension (str): File extension to filter and apply
        start_number (int): Starting number for sequential naming
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
    
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    if not files:
        print(f"No files found in directory '{directory}'.")
        return
    
    filtered_files = [f for f in files if f.endswith(extension)]
    
    if not filtered_files:
        print(f"No files with extension '{extension}' found in directory '{directory}'.")
        return
    
    filtered_files.sort()
    
    renamed_count = 0
    for index, filename in enumerate(filtered_files, start=start_number):
        old_path = os.path.join(directory, filename)
        new_filename = f"{prefix}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: '{filename}' -> '{new_filename}'")
            renamed_count += 1
        except OSError as e:
            print(f"Error renaming '{filename}': {e}")
    
    print(f"\nSuccessfully renamed {renamed_count} file(s).")

def main():
    parser = argparse.ArgumentParser(description="Rename files with sequential numbering.")
    parser.add_argument("directory", help="Directory containing files to rename")
    parser.add_argument("-p", "--prefix", default="file", help="Prefix for renamed files (default: file)")
    parser.add_argument("-e", "--extension", default=".txt", help="File extension to filter (default: .txt)")
    parser.add_argument("-s", "--start", type=int, default=1, help="Starting number (default: 1)")
    
    args = parser.parse_args()
    
    rename_files_with_sequential_numbering(
        directory=args.directory,
        prefix=args.prefix,
        extension=args.extension,
        start_number=args.start
    )

if __name__ == "__main__":
    main()