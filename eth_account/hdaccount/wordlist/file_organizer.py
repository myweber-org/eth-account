
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    path = Path(directory_path)
    
    # Define categories and their associated extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.json'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    }

    # Create category folders if they don't exist
    for category in categories.keys():
        category_path = path / category
        category_path.mkdir(exist_ok=True)

    # Track moved files count
    moved_files = 0

    # Iterate over files in the directory
    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False

            # Find the appropriate category for the file
            for category, extensions in categories.items():
                if file_extension in extensions:
                    destination = path / category / item.name
                    
                    # Handle naming conflicts
                    counter = 1
                    while destination.exists():
                        stem = item.stem
                        new_name = f"{stem}_{counter}{item.suffix}"
                        destination = path / category / new_name
                        counter += 1
                    
                    shutil.move(str(item), str(destination))
                    moved_files += 1
                    moved = True
                    break

            # If no category matches, move to 'Other' folder
            if not moved:
                other_folder = path / 'Other'
                other_folder.mkdir(exist_ok=True)
                destination = other_folder / item.name
                
                # Handle naming conflicts for 'Other' files
                counter = 1
                while destination.exists():
                    stem = item.stem
                    new_name = f"{stem}_{counter}{item.suffix}"
                    destination = other_folder / new_name
                    counter += 1
                
                shutil.move(str(item), str(destination))
                moved_files += 1

    print(f"Organization complete. Moved {moved_files} files.")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
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
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }

    # Convert directory to Path object for easier handling
    base_path = Path(directory)

    # Check if the directory exists
    if not base_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Create category folders if they don't exist
    for category in file_categories.keys():
        category_path = base_path / category
        category_path.mkdir(exist_ok=True)

    # Track moved files and errors
    moved_files = []
    error_files = []

    # Iterate through all items in the directory
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
            error_files.append((item.name, "File already exists in target location"))
            continue

        try:
            # Move the file
            shutil.move(str(item), str(target_path))
            moved_files.append((item.name, target_category))
        except Exception as e:
            error_files.append((item.name, str(e)))

    # Print summary
    print(f"\nOrganization complete for: {directory}")
    print(f"Files moved: {len(moved_files)}")
    
    if moved_files:
        print("\nMoved files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if error_files:
        print(f"\nErrors: {len(error_files)}")
        for filename, error_msg in error_files:
            print(f"  {filename}: {error_msg}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_directory = os.getcwd()
    organize_files(current_directory)