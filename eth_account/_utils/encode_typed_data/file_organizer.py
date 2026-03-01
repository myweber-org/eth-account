
import os
import shutil
from pathlib import Path

def organize_files(source_dir, organize_by='extension'):
    """
    Organizes files in the source directory into subfolders based on criteria.
    Default is to organize by file extension.
    """
    source_path = Path(source_dir)
    
    if not source_path.exists() or not source_path.is_dir():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return
    
    for item in source_path.iterdir():
        if item.is_file():
            if organize_by == 'extension':
                # Get file extension, use 'no_extension' for files without one
                suffix = item.suffix.lower()
                if suffix:
                    folder_name = suffix[1:]  # Remove the dot
                else:
                    folder_name = 'no_extension'
            else:
                # For other organization methods (placeholder for future expansion)
                folder_name = 'other'
            
            target_folder = source_path / folder_name
            target_folder.mkdir(exist_ok=True)
            
            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    # Example usage: organize files on the desktop
    desktop_path = Path.home() / 'Desktop'
    if desktop_path.exists():
        organize_files(desktop_path)
        print("File organization complete.")
    else:
        print("Desktop directory not found.")
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return
    
    path = Path(directory_path)
    
    extension_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.rar', '.tar', '.gz'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
        'Executables': ['.exe', '.msi', '.app', '.sh']
    }
    
    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            destination_folder = None
            
            for category, extensions in extension_categories.items():
                if file_extension in extensions:
                    destination_folder = category
                    break
            
            if not destination_folder:
                destination_folder = 'Other'
            
            destination_path = path / destination_folder
            destination_path.mkdir(exist_ok=True)
            
            try:
                shutil.move(str(item), str(destination_path / item.name))
                print(f"Moved: {item.name} -> {destination_folder}/")
            except Exception as e:
                print(f"Error moving {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)