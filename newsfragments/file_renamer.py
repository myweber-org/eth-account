
import os
import sys
from datetime import datetime

def rename_files_with_timestamp(directory):
    try:
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a valid directory")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        renamed_count = 0

        for filename in os.listdir(directory):
            old_path = os.path.join(directory, filename)
            
            if os.path.isfile(old_path):
                name, ext = os.path.splitext(filename)
                new_filename = f"{timestamp}_{name}{ext}"
                new_path = os.path.join(directory, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"Failed to rename {filename}: {e}")

        print(f"Total files renamed: {renamed_count}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_renamer.py <directory_path>")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    rename_files_with_timestamp(target_directory)