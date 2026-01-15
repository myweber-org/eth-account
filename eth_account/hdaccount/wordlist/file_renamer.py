
import os
import sys
from datetime import datetime

def rename_files_by_date(directory, prefix="file_"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        for filename in files:
            filepath = os.path.join(directory, filename)
            stat = os.stat(filepath)
            creation_time = datetime.fromtimestamp(stat.st_ctime)
            
            new_name = f"{prefix}{creation_time.strftime('%Y%m%d_%H%M%S')}{os.path.splitext(filename)[1]}"
            new_path = os.path.join(directory, new_name)
            
            counter = 1
            while os.path.exists(new_path):
                base, ext = os.path.splitext(new_name)
                new_name = f"{base}_{counter}{ext}"
                new_path = os.path.join(directory, new_name)
                counter += 1
            
            os.rename(filepath, new_path)
            print(f"Renamed: {filename} -> {new_name}")
            
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "file_"
    
    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a valid directory")
        sys.exit(1)
    
    success = rename_files_by_date(dir_path, prefix)
    sys.exit(0 if success else 1)