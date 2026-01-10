
import os
import sys
from datetime import datetime

def rename_files(directory, prefix):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        for filename in files:
            filepath = os.path.join(directory, filename)
            creation_time = os.path.getctime(filepath)
            date_str = datetime.fromtimestamp(creation_time).strftime('%Y%m%d_%H%M%S')
            
            file_ext = os.path.splitext(filename)[1]
            new_filename = f"{prefix}_{date_str}{file_ext}"
            new_filepath = os.path.join(directory, new_filename)
            
            counter = 1
            while os.path.exists(new_filepath):
                new_filename = f"{prefix}_{date_str}_{counter}{file_ext}"
                new_filepath = os.path.join(directory, new_filename)
                counter += 1
            
            os.rename(filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")
            
        print(f"Successfully renamed {len(files)} files.")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python file_renamer.py <directory_path> <prefix>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    name_prefix = sys.argv[2]
    
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a valid directory.")
        sys.exit(1)
    
    rename_files(target_dir, name_prefix)