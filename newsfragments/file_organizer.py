
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define file type categories
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }
    
    # Ensure the directory exists
    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    # Create category folders if they don't exist
    for category in categories:
        (target_dir / category).mkdir(exist_ok=True)
    
    # Create an 'Other' folder for uncategorized files
    other_dir = target_dir / 'Other'
    other_dir.mkdir(exist_ok=True)
    
    moved_count = 0
    skipped_count = 0
    
    # Iterate through files in the directory
    for item in target_dir.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            moved = False
            
            # Find the appropriate category
            for category, extensions in categories.items():
                if file_ext in extensions:
                    dest_dir = target_dir / category
                    try:
                        shutil.move(str(item), str(dest_dir / item.name))
                        moved_count += 1
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")
            
            # If file doesn't match any category, move to 'Other'
            if not moved:
                try:
                    shutil.move(str(item), str(other_dir / item.name))
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {item.name} to Other: {e}")
                    skipped_count += 1
    
    print(f"Organization complete. Moved {moved_count} files, skipped {skipped_count}.")

if __name__ == "__main__":
    # Get directory from user input or use current directory
    import sys
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    else:
        target_directory = os.getcwd()
    
    print(f"Organizing files in: {target_directory}")
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    base_path = Path(directory).resolve()

    # Define category mappings for common extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp"],
        "Documents": [".pdf", ".docx", ".txt", ".md", ".rtf", ".odt", ".xlsx", ".pptx"],
        "Audio": [".mp3", ".wav", ".flac", ".aac", ".ogg"],
        "Video": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"],
        "Archives": [".zip", ".rar", ".7z", ".tar", ".gz"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".json", ".xml"],
        "Executables": [".exe", ".msi", ".sh", ".bat", ".app"]
    }

    # Create a reverse lookup: extension -> category
    ext_to_category = {}
    for category, extensions in categories.items():
        for ext in extensions:
            ext_to_category[ext.lower()] = category

    # Ensure the directory exists
    if not base_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Iterate over items in the directory
    for item in base_path.iterdir():
        # Skip directories
        if item.is_dir():
            continue

        # Get file extension
        ext = item.suffix.lower()

        # Determine category
        category = ext_to_category.get(ext, "Other")

        # Create category folder if it doesn't exist
        category_folder = base_path / category
        category_folder.mkdir(exist_ok=True)

        # Move the file
        try:
            dest_path = category_folder / item.name
            # Handle name conflicts
            counter = 1
            while dest_path.exists():
                stem = item.stem
                new_name = f"{stem}_{counter}{item.suffix}"
                dest_path = category_folder / new_name
                counter += 1

            shutil.move(str(item), str(dest_path))
            print(f"Moved: {item.name} -> {category}/{dest_path.name}")
        except Exception as e:
            print(f"Failed to move {item.name}: {e}")

    print("File organization complete.")

if __name__ == "__main__":
    # You can specify a directory as a command-line argument
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    organize_files(target_dir)