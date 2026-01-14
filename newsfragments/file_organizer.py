
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define file type categories and their associated extensions
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv']
    }

    # Convert to Path object for easier handling
    base_path = Path(directory_path)

    # Ensure the directory exists
    if not base_path.is_dir():
        print(f"Error: The directory '{directory_path}' does not exist.")
        return

    # Create category folders if they don't exist
    for category in file_categories.keys():
        category_folder = base_path / category
        category_folder.mkdir(exist_ok=True)

    # Track moved files and errors
    moved_files = []
    errors = []

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

        # If no category found, skip or move to 'Other'
        if target_category is None:
            target_category = 'Other'
            other_folder = base_path / target_category
            other_folder.mkdir(exist_ok=True)

        # Define target path
        target_folder = base_path / target_category
        target_path = target_folder / item.name

        # Move the file
        try:
            # Handle name conflicts
            if target_path.exists():
                # Add a number to the filename before the extension
                counter = 1
                while target_path.exists():
                    new_name = f"{item.stem}_{counter}{item.suffix}"
                    target_path = target_folder / new_name
                    counter += 1

            shutil.move(str(item), str(target_path))
            moved_files.append((item.name, target_category))
        except Exception as e:
            errors.append((item.name, str(e)))

    # Print summary
    print(f"Organization complete for: {directory_path}")
    print(f"Files moved: {len(moved_files)}")
    
    if moved_files:
        print("\nMoved files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for filename, error_msg in errors:
            print(f"  {filename}: {error_msg}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_directory = os.getcwd()
    organize_files(current_directory)