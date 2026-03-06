
import os
import shutil

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_dir = os.path.join(directory, file_extension)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved {filename} to {file_extension}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subdirectories based on their file extensions.
    """
    base_path = Path(directory).resolve()
    
    if not base_path.is_dir():
        print(f"Error: '{directory}' is not a valid directory.")
        return

    extension_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.md', '.xlsx', '.pptx'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mkv', '.mov']
    }

    for item in base_path.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            target_category = 'Other'

            for category, extensions in extension_categories.items():
                if file_ext in extensions:
                    target_category = category
                    break

            target_dir = base_path / target_category
            target_dir.mkdir(exist_ok=True)

            try:
                shutil.move(str(item), str(target_dir / item.name))
                print(f"Moved: {item.name} -> {target_category}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    organize_files()