
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    base_path = Path(directory_path)
    
    # Define categories and their associated extensions
    categories = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'audio': ['.mp3', '.wav', '.flac', '.aac'],
        'video': ['.mp4', '.avi', '.mov', '.mkv'],
        'archives': ['.zip', '.rar', '.tar', '.gz'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
    }

    # Create category folders if they don't exist
    for category in categories:
        category_folder = base_path / category
        category_folder.mkdir(exist_ok=True)

    # Track moved files
    moved_files = []

    # Iterate through all items in the directory
    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False

            # Find the appropriate category for the file
            for category, extensions in categories.items():
                if file_extension in extensions:
                    target_folder = base_path / category
                    try:
                        shutil.move(str(item), str(target_folder / item.name))
                        moved_files.append((item.name, category))
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")

            # If no category matched, move to 'others'
            if not moved:
                others_folder = base_path / 'others'
                others_folder.mkdir(exist_ok=True)
                try:
                    shutil.move(str(item), str(others_folder / item.name))
                    moved_files.append((item.name, 'others'))
                except Exception as e:
                    print(f"Error moving {item.name} to others: {e}")

    # Print summary
    if moved_files:
        print(f"Organized {len(moved_files)} file(s):")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    else:
        print("No files were moved.")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)