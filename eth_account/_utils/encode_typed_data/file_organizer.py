
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organize files in the given directory by their extensions.
    Creates subdirectories for each file type and moves files accordingly.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    path = Path(directory_path)
    
    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            if file_extension:
                target_dir = path / file_extension[1:]  
            else:
                target_dir = path / "no_extension"
            
            target_dir.mkdir(exist_ok=True)
            
            try:
                shutil.move(str(item), str(target_dir / item.name))
                print(f"Moved: {item.name} -> {target_dir.name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into folders
    named after their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix[1:].lower()
            
            if not file_extension:
                file_extension = "no_extension"
            
            target_folder = os.path.join(directory, file_extension)
            
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            
            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {file_extension}/")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)