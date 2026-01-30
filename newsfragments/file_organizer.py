
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