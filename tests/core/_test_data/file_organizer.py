
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

            if not file_extension:
                folder_name = "NoExtension"
            else:
                folder_name = file_extension[1:].capitalize() + "Files"

            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_dir = input("Enter the directory path to organize: ").strip()
    organize_files(target_dir)
    print("File organization complete.")
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define file type categories and their associated extensions
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c']
    }

    # Convert directory to Path object for easier handling
    base_path = Path(directory)

    # Ensure the directory exists
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return

    # Create category folders if they don't exist
    for category in file_categories.keys():
        category_path = base_path / category
        category_path.mkdir(exist_ok=True)

    # Track moved files and errors
    moved_files = []
    error_files = []

    # Iterate over all items in the directory
    for item in base_path.iterdir():
        # Skip directories
        if item.is_dir():
            continue

        # Get file extension
        file_extension = item.suffix.lower()

        # Find the appropriate category for the file
        target_category = None
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # If no category found, skip the file
        if not target_category:
            continue

        # Define target path
        target_path = base_path / target_category / item.name

        # Check if file already exists in target location
        if target_path.exists():
            error_files.append((item.name, "File already exists in target folder"))
            continue

        try:
            # Move the file
            shutil.move(str(item), str(target_path))
            moved_files.append((item.name, target_category))
        except Exception as e:
            error_files.append((item.name, str(e)))

    # Print summary
    print(f"Organization complete for: {directory}")
    print(f"Files moved: {len(moved_files)}")
    
    if moved_files:
        print("\nMoved files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if error_files:
        print(f"\nErrors: {len(error_files)}")
        for filename, error in error_files:
            print(f"  {filename}: {error}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    target_directory = os.getcwd()
    organize_files(target_directory)