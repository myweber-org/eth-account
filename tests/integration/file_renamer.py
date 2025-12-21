
import os
import sys
from datetime import datetime

def rename_files_by_date(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        mod_time = os.path.getmtime(filepath)
        date_str = datetime.fromtimestamp(mod_time).strftime("%Y%m%d_%H%M%S")
        
        name, ext = os.path.splitext(filename)
        new_filename = f"{date_str}{ext}"
        new_filepath = os.path.join(directory, new_filename)
        
        counter = 1
        while os.path.exists(new_filepath):
            new_filename = f"{date_str}_{counter}{ext}"
            new_filepath = os.path.join(directory, new_filename)
            counter += 1
        
        os.rename(filepath, new_filepath)
        print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_renamer.py <directory_path>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    rename_files_by_date(target_dir)