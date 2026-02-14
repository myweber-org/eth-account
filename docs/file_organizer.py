
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organizes files in the specified directory by moving them into
    subdirectories named after their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                target_folder = os.path.join(directory_path, file_extension[1:] + "_files")
            else:
                target_folder = os.path.join(directory_path, "no_extension_files")
            
            os.makedirs(target_folder, exist_ok=True)
            
            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {target_folder}")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the given directory by moving them into
    subfolders based on their file extensions.
    """
    # Define file type categories and their associated extensions
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
    }

    # Convert the directory path to a Path object for easier handling
    path = Path(directory_path)

    # Check if the provided path exists and is a directory
    if not path.exists() or not path.is_dir():
        print(f"Error: The path '{directory_path}' does not exist or is not a directory.")
        return

    # Iterate over all items in the directory
    for item in path.iterdir():
        # Skip if it's a directory (we only want to organize files)
        if item.is_dir():
            continue

        # Get the file extension (lowercase for case-insensitive matching)
        file_extension = item.suffix.lower()

        # Determine the target category folder
        target_category = 'Other'  # Default category for unmatched extensions
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # Create the target category folder if it doesn't exist
        target_folder = path / target_category
        target_folder.mkdir(exist_ok=True)

        # Construct the destination path
        destination = target_folder / item.name

        # Check if a file with the same name already exists in the target folder
        if destination.exists():
            print(f"Warning: '{item.name}' already exists in '{target_category}'. Skipping.")
            continue

        # Move the file to the target folder
        try:
            shutil.move(str(item), str(destination))
            print(f"Moved: {item.name} -> {target_category}/")
        except Exception as e:
            print(f"Error moving {item.name}: {e}")

if __name__ == "__main__":
    # Example usage: organize files in the current working directory
    current_directory = os.getcwd()
    organize_files(current_directory)
    print("File organization complete.")
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into folders
    based on their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            folder_path = os.path.join(directory, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            destination_path = os.path.join(folder_path, item)
            
            if not os.path.exists(destination_path):
                shutil.move(item_path, destination_path)
                print(f"Moved: {item} -> {folder_name}/")
            else:
                print(f"Skipped: {item} already exists in {folder_name}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization complete.")
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subfolders
    named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()

            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"

            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            try:
                shutil.move(item_path, target_folder)
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)