
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

def organize_files(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    categories = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'audio': ['.mp3', '.wav', '.flac', '.aac'],
        'video': ['.mp4', '.avi', '.mov', '.mkv'],
        'archives': ['.zip', '.rar', '.7z', '.tar.gz'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if os.path.isfile(file_path):
            file_ext = Path(filename).suffix.lower()
            moved = False
            
            for category, extensions in categories.items():
                if file_ext in extensions:
                    category_dir = os.path.join(directory_path, category)
                    os.makedirs(category_dir, exist_ok=True)
                    
                    destination = os.path.join(category_dir, filename)
                    shutil.move(file_path, destination)
                    print(f"Moved: {filename} -> {category}/")
                    moved = True
                    break
            
            if not moved:
                other_dir = os.path.join(directory_path, 'other')
                os.makedirs(other_dir, exist_ok=True)
                shutil.move(file_path, os.path.join(other_dir, filename))
                print(f"Moved: {filename} -> other/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)