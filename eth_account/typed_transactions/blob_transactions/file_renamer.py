
import os
import glob
from pathlib import Path
from datetime import datetime

def rename_files_sequential(directory, prefix="file", extension=".txt"):
    files = glob.glob(os.path.join(directory, "*" + extension))
    files.sort(key=os.path.getctime)
    
    for index, filepath in enumerate(files, start=1):
        creation_time = datetime.fromtimestamp(os.path.getctime(filepath))
        timestamp = creation_time.strftime("%Y%m%d_%H%M%S")
        new_filename = f"{prefix}_{timestamp}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_filename)
        
        try:
            os.rename(filepath, new_path)
            print(f"Renamed: {Path(filepath).name} -> {new_filename}")
        except OSError as e:
            print(f"Error renaming {filepath}: {e}")

if __name__ == "__main__":
    target_dir = input("Enter directory path: ").strip()
    if os.path.isdir(target_dir):
        rename_files_sequential(target_dir)
    else:
        print("Invalid directory path.")