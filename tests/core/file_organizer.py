
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organize files in the given directory by moving them into folders
    named after their file extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            target_folder = os.path.join(directory_path, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization complete.")
import os
import shutil

def organize_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            extension = extension.lower()

            if extension:
                folder_name = extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"

            target_folder = os.path.join(directory, folder_name)

            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            try:
                shutil.move(file_path, os.path.join(target_folder, filename))
                print(f"Moved: {filename} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {filename}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the given directory by moving them into
    subfolders named after their file extensions.
    """
    path = Path(directory_path)

    if not path.exists() or not path.is_dir():
        print(f"Error: The directory '{directory_path}' does not exist.")
        return

    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            if file_extension:
                folder_name = file_extension[1:]  # Remove the leading dot
            else:
                folder_name = "no_extension"

            target_folder = path / folder_name
            target_folder.mkdir(exist_ok=True)

            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)