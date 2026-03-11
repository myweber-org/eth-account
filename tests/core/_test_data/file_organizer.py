
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organizes files in the specified directory into subfolders based on their file extensions.
    """
    # Convert the input path to a Path object for easier handling
    base_path = Path(directory_path)

    # Check if the provided path exists and is a directory
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: The path '{directory_path}' does not exist or is not a directory.")
        return

    # Dictionary mapping file extensions to folder names
    extension_categories = {
        '.txt': 'TextFiles',
        '.pdf': 'PDFs',
        '.jpg': 'Images',
        '.jpeg': 'Images',
        '.png': 'Images',
        '.gif': 'Images',
        '.mp3': 'Audio',
        '.wav': 'Audio',
        '.mp4': 'Videos',
        '.avi': 'Videos',
        '.mov': 'Videos',
        '.zip': 'Archives',
        '.tar': 'Archives',
        '.gz': 'Archives',
        '.py': 'PythonScripts',
        '.js': 'Scripts',
        '.html': 'WebFiles',
        '.css': 'WebFiles',
        '.json': 'Data',
        '.csv': 'Data',
        '.xlsx': 'Spreadsheets',
        '.docx': 'Documents',
        '.pptx': 'Presentations',
    }

    # Iterate over all items in the directory
    for item in base_path.iterdir():
        # Skip if it's a directory
        if item.is_dir():
            continue

        # Get the file extension (lowercase for consistency)
        file_extension = item.suffix.lower()

        # Determine the target folder name
        if file_extension in extension_categories:
            target_folder_name = extension_categories[file_extension]
        else:
            target_folder_name = 'Other'

        # Create the full path for the target folder
        target_folder_path = base_path / target_folder_name

        # Create the target folder if it doesn't exist
        target_folder_path.mkdir(exist_ok=True)

        # Construct the new file path
        new_file_path = target_folder_path / item.name

        # Move the file to the target folder
        try:
            shutil.move(str(item), str(new_file_path))
            print(f"Moved: {item.name} -> {target_folder_name}/")
        except Exception as e:
            print(f"Failed to move {item.name}: {e}")

    print("File organization complete.")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    target_directory = input("Enter the directory path to organize (or press Enter for current directory): ").strip()
    
    if not target_directory:
        target_directory = os.getcwd()
    
    organize_files_by_extension(target_directory)
import os
import shutil

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subdirectories
    based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'NoExtension'
            target_dir = os.path.join(directory, file_extension.upper())

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            try:
                shutil.move(file_path, os.path.join(target_dir, filename))
                print(f"Moved: {filename} -> {target_dir}")
            except Exception as e:
                print(f"Failed to move {filename}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into
    subfolders based on their file extensions.
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
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil

def organize_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_folder = os.path.join(directory, file_extension.upper() + "_FILES")

            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            shutil.move(file_path, os.path.join(target_folder, filename))
            print(f"Moved: {filename} -> {target_folder}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)