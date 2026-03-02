
import os
import sys

def rename_files_sequentially(directory, prefix="file", extension=".txt"):
    """
    Rename all files in the given directory with sequential numbering.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False
    
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()
    
    for index, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        new_filename = f"{prefix}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
        except OSError as e:
            print(f"Error renaming {filename}: {e}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [extension]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    extension = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    rename_files_sequentially(dir_path, prefix, extension)
import os
import sys
from datetime import datetime

def rename_files_in_directory(directory_path, prefix="file"):
    try:
        if not os.path.isdir(directory_path):
            print(f"Error: {directory_path} is not a valid directory.")
            return False

        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        renamed_count = 0

        for filename in files:
            file_path = os.path.join(directory_path, filename)
            creation_time = os.path.getctime(file_path)
            date_str = datetime.fromtimestamp(creation_time).strftime("%Y%m%d_%H%M%S")
            name, extension = os.path.splitext(filename)
            new_filename = f"{prefix}_{date_str}{extension}"
            new_file_path = os.path.join(directory_path, new_filename)

            if not os.path.exists(new_file_path):
                os.rename(file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            else:
                print(f"Skipped: {new_filename} already exists.")

        print(f"Renaming complete. {renamed_count} files renamed.")
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix]")
        sys.exit(1)

    dir_path = sys.argv[1]
    user_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    rename_files_in_directory(dir_path, user_prefix)