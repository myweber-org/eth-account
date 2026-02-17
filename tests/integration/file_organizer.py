
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
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
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if not file_extension:
                folder_name = "no_extension"
            else:
                folder_name = file_extension[1:] + "_files"
            
            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            target_path = os.path.join(target_folder, item)
            
            if not os.path.exists(target_path):
                shutil.move(item_path, target_path)
                print(f"Moved: {item} -> {folder_name}/")
            else:
                print(f"Skipped: {item} already exists in {folder_name}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    base_path = Path(directory_path)
    
    extension_mapping = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'audio': ['.mp3', '.wav', '.flac', '.aac'],
        'video': ['.mp4', '.avi', '.mov', '.mkv'],
        'archives': ['.zip', '.rar', '.tar', '.gz'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }

    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            target_folder = None

            for folder, extensions in extension_mapping.items():
                if file_extension in extensions:
                    target_folder = folder
                    break

            if not target_folder:
                target_folder = 'others'

            target_path = base_path / target_folder
            target_path.mkdir(exist_ok=True)

            try:
                shutil.move(str(item), str(target_path / item.name))
                print(f"Moved: {item.name} -> {target_folder}/")
            except Exception as e:
                print(f"Error moving {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil

def organize_files(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if os.path.isfile(file_path):
            _, file_extension = os.path.splitext(filename)
            extension = file_extension.lower()[1:] if file_extension else "no_extension"

            target_folder = os.path.join(directory_path, extension)
            os.makedirs(target_folder, exist_ok=True)

            target_path = os.path.join(target_folder, filename)
            shutil.move(file_path, target_path)
            print(f"Moved: {filename} -> {extension}/")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organizes files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
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
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
import os
import shutil

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_folder = os.path.join(directory, file_extension.upper() + "_FILES")

            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            target_path = os.path.join(target_folder, filename)
            shutil.move(file_path, target_path)
            print(f"Moved {filename} to {target_folder}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)