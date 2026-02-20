
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