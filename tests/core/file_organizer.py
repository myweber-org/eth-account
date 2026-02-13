
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
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Define categories and their associated extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv']
    }

    # Create category folders if they don't exist
    for category in categories:
        category_path = os.path.join(directory, category)
        os.makedirs(category_path, exist_ok=True)

    # Track moved files and errors
    moved_files = []
    errors = []

    # Iterate over files in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        # Skip directories
        if os.path.isdir(item_path):
            continue

        # Get file extension
        file_extension = Path(item).suffix.lower()

        # Find the appropriate category
        target_category = None
        for category, extensions in categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # If no category found, use 'Other'
        if target_category is None:
            target_category = 'Other'
            other_path = os.path.join(directory, target_category)
            os.makedirs(other_path, exist_ok=True)

        # Move the file
        try:
            target_path = os.path.join(directory, target_category, item)
            shutil.move(item_path, target_path)
            moved_files.append((item, target_category))
        except Exception as e:
            errors.append((item, str(e)))

    # Print summary
    print(f"Organization complete for: {directory}")
    print(f"Files moved: {len(moved_files)}")
    if moved_files:
        print("\nMoved files:")
        for file_name, category in moved_files:
            print(f"  {file_name} -> {category}/")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for file_name, error_msg in errors:
            print(f"  {file_name}: {error_msg}")

if __name__ == "__main__":
    # Use current directory if no argument provided
    target_dir = input("Enter directory path to organize (or press Enter for current): ").strip()
    if not target_dir:
        target_dir = os.getcwd()
    
    organize_files(target_dir)
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subdirectories based on their file extensions.
    """
    base_path = Path(directory).resolve()
    
    if not base_path.is_dir():
        print(f"Error: '{directory}' is not a valid directory.")
        return

    extension_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.json', '.xml'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv']
    }

    other_dir = base_path / 'Other'
    other_dir.mkdir(exist_ok=True)

    for item in base_path.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            moved = False

            for category, extensions in extension_categories.items():
                if file_ext in extensions:
                    category_dir = base_path / category
                    category_dir.mkdir(exist_ok=True)
                    try:
                        shutil.move(str(item), str(category_dir / item.name))
                        print(f"Moved: {item.name} -> {category}/")
                        moved = True
                        break
                    except Exception as e:
                        print(f"Failed to move {item.name}: {e}")

            if not moved:
                try:
                    shutil.move(str(item), str(other_dir / item.name))
                    print(f"Moved: {item.name} -> Other/")
                except Exception as e:
                    print(f"Failed to move {item.name}: {e}")

    print("\nFile organization completed.")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize (or press Enter for current): ").strip()
    if not target_directory:
        target_directory = "."
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.rar', '.tar', '.gz'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }
    
    for category in categories:
        category_path = os.path.join(directory, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isdir(item_path):
            continue
        
        file_extension = Path(item).suffix.lower()
        moved = False
        
        for category, extensions in categories.items():
            if file_extension in extensions:
                dest_path = os.path.join(directory, category, item)
                shutil.move(item_path, dest_path)
                print(f"Moved '{item}' to '{category}' folder.")
                moved = True
                break
        
        if not moved:
            other_path = os.path.join(directory, 'Other')
            if not os.path.exists(other_path):
                os.makedirs(other_path)
            dest_path = os.path.join(other_path, item)
            shutil.move(item_path, dest_path)
            print(f"Moved '{item}' to 'Other' folder.")
    
    print("File organization completed.")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)