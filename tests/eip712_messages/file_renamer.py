import os
import glob
from pathlib import Path

def rename_files_sequential(directory, prefix="file", extension=".txt"):
    files = sorted(Path(directory).iterdir(), key=os.path.getctime)
    count = 1
    
    for file_path in files:
        if file_path.is_file():
            new_name = f"{prefix}_{count:03d}{extension}"
            new_path = file_path.parent / new_name
            file_path.rename(new_path)
            print(f"Renamed: {file_path.name} -> {new_name}")
            count += 1

if __name__ == "__main__":
    target_dir = "./documents"
    if Path(target_dir).exists():
        rename_files_sequential(target_dir, "document", ".pdf")
    else:
        print(f"Directory '{target_dir}' not found.")