
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the given directory by moving them into
    subfolders based on their file extensions.
    """
    # Define categories and their associated extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md"],
        "Audio": [".mp3", ".wav", ".flac", ".aac"],
        "Video": [".mp4", ".mkv", ".avi", ".mov"],
        "Archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c"],
    }

    # Ensure the directory path is a Path object
    base_path = Path(directory_path)

    # Check if the directory exists
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: The directory '{directory_path}' does not exist or is not a directory.")
        return

    # Create category folders if they don't exist
    for category in categories:
        category_path = base_path / category
        category_path.mkdir(exist_ok=True)

    # Create an 'Other' folder for uncategorized files
    other_path = base_path / "Other"
    other_path.mkdir(exist_ok=True)

    # Track moved files
    moved_files = []

    # Iterate over all items in the directory
    for item in base_path.iterdir():
        # Skip directories (including the ones we just created)
        if item.is_dir():
            continue

        # Get the file extension
        extension = item.suffix.lower()

        # Determine the target category
        target_category = None
        for category, extensions in categories.items():
            if extension in extensions:
                target_category = category
                break

        # If no category found, use 'Other'
        if target_category is None:
            target_category = "Other"

        # Define the target path
        target_path = base_path / target_category / item.name

        # Move the file, handling name conflicts
        try:
            if target_path.exists():
                # Append a number to avoid overwriting
                counter = 1
                name_stem = item.stem
                while target_path.exists():
                    new_name = f"{name_stem}_{counter}{item.suffix}"
                    target_path = base_path / target_category / new_name
                    counter += 1

            shutil.move(str(item), str(target_path))
            moved_files.append((item.name, target_category))
        except Exception as e:
            print(f"Failed to move '{item.name}': {e}")

    # Print summary
    if moved_files:
        print(f"Organized {len(moved_files)} file(s) in '{directory_path}':")
        for file_name, category in moved_files:
            print(f"  - {file_name} -> {category}/")
    else:
        print("No files were moved.")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    target_directory = input("Enter the directory path to organize (or press Enter for current directory): ").strip()
    if not target_directory:
        target_directory = "."

    organize_files(target_directory)
import os
import shutil

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
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
            
            folder_path = os.path.join(directory, folder_name)
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            shutil.move(file_path, os.path.join(folder_path, filename))
            print(f"Moved {filename} to {folder_name}/")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """Organize files in the specified directory by their extensions."""
    base_path = Path(directory).resolve()
    
    # Define categories and their associated file extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".rtf"],
        "Archives": [".zip", ".tar", ".gz", ".7z", ".rar"],
        "Audio": [".mp3", ".wav", ".flac", ".aac", ".ogg"],
        "Video": [".mp4", ".avi", ".mkv", ".mov", ".wmv"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".json"],
        "Executables": [".exe", ".msi", ".sh", ".bat", ".app"],
    }
    
    # Create a default 'Others' category for unclassified extensions
    others_category = "Others"
    
    # Track moved files for reporting
    moved_files = []
    
    # Ensure the directory exists
    if not base_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    # Create category directories if they don't exist
    for category in list(categories.keys()) + [others_category]:
        category_dir = base_path / category
        category_dir.mkdir(exist_ok=True)
    
    # Iterate over items in the directory
    for item in base_path.iterdir():
        # Skip directories and hidden files
        if item.is_dir() or item.name.startswith('.'):
            continue
        
        # Determine the category based on file extension
        file_ext = item.suffix.lower()
        target_category = others_category
        
        for category, extensions in categories.items():
            if file_ext in extensions:
                target_category = category
                break
        
        # Define target path
        target_dir = base_path / target_category
        target_path = target_dir / item.name
        
        # Handle name conflicts by appending a number
        counter = 1
        while target_path.exists():
            stem = item.stem
            new_name = f"{stem}_{counter}{item.suffix}"
            target_path = target_dir / new_name
            counter += 1
        
        # Move the file
        try:
            shutil.move(str(item), str(target_path))
            moved_files.append((item.name, target_category))
        except Exception as e:
            print(f"Failed to move '{item.name}': {e}")
    
    # Print summary
    if moved_files:
        print(f"Organized {len(moved_files)} file(s) in '{base_path}':")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    else:
        print("No files were moved.")

if __name__ == "__main__":
    # Example: organize files in the current directory
    organize_files()