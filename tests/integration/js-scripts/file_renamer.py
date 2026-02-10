
import os
import sys
from datetime import datetime

def rename_files_by_date(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        return False
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if os.path.isfile(filepath):
            try:
                creation_time = os.path.getctime(filepath)
                date_str = datetime.fromtimestamp(creation_time).strftime("%Y%m%d_%H%M%S")
                
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
                
            except Exception as e:
                print(f"Failed to rename {filename}: {e}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_renamer.py <directory_path>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    rename_files_by_date(target_dir)
import os
import sys

def rename_files_with_sequence(directory, prefix="file"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            extension = os.path.splitext(filename)[1]
            new_name = f"{prefix}_{index:03d}{extension}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
        
        print(f"Successfully renamed {len(files)} files.")
        return True
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix]")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    name_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a valid directory.")
        sys.exit(1)
    
    rename_files_with_sequence(target_dir, name_prefix)