
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            file_ext = Path(item).suffix.lower()
            if file_ext:
                target_folder = os.path.join(directory, file_ext[1:] + "_files")
            else:
                target_folder = os.path.join(directory, "no_extension_files")

            os.makedirs(target_folder, exist_ok=True)
            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {target_folder}")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_dir = input("Enter the directory path to organize: ").strip()
    organize_files(target_dir)