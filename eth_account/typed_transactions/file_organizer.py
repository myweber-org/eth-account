
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
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
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
            
            target_path = os.path.join(target_folder, item)
            
            if not os.path.exists(target_path):
                shutil.move(item_path, target_path)
                print(f"Moved: {item} -> {folder_name}/")
            else:
                print(f"Skipped: {item} (already exists in {folder_name})")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)