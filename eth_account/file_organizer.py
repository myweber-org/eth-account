
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organize files in the specified directory by their extensions.
    Creates folders for each file type and moves files accordingly.
    """
    base_path = Path(directory).resolve()
    
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return
    
    extension_map = {}
    
    for item in base_path.iterdir():
        if item.is_file():
            ext = item.suffix.lower()
            if ext:
                ext = ext[1:]
            else:
                ext = "no_extension"
            
            if ext not in extension_map:
                extension_map[ext] = []
            extension_map[ext].append(item)
    
    for ext, files in extension_map.items():
        target_dir = base_path / ext
        target_dir.mkdir(exist_ok=True)
        
        for file_path in files:
            try:
                shutil.move(str(file_path), str(target_dir / file_path.name))
                print(f"Moved: {file_path.name} -> {ext}/")
            except Exception as e:
                print(f"Failed to move {file_path.name}: {e}")
    
    print(f"\nOrganization complete. Created {len(extension_map)} category folders.")

if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    organize_files(target_dir)
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    base_path = Path(directory_path)

    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            if file_extension:
                target_folder_name = file_extension[1:] + "_files"
            else:
                target_folder_name = "no_extension_files"

            target_folder = base_path / target_folder_name
            target_folder.mkdir(exist_ok=True)

            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {target_folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)