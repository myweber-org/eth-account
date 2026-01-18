
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