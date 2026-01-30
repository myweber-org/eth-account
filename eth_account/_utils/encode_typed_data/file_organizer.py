
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