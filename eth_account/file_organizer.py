
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organize files in the specified directory by their extensions.
    Creates folders for each file type and moves files accordingly.
    """
    base_path = Path(directory).resolve()
    
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return
    
    extension_map = {}
    
    for item in base_path.iterdir():
        if item.is_file():
            ext = item.suffix.lower()
            if ext:
                ext = ext[1:]
            else:
                ext = "no_extension"
            
            if ext not in extension_map:
                extension_map[ext] = []
            extension_map[ext].append(item)
    
    for ext, files in extension_map.items():
        target_dir = base_path / ext
        target_dir.mkdir(exist_ok=True)
        
        for file_path in files:
            try:
                shutil.move(str(file_path), str(target_dir / file_path.name))
                print(f"Moved: {file_path.name} -> {ext}/")
            except Exception as e:
                print(f"Failed to move {file_path.name}: {e}")
    
    print(f"\nOrganization complete. Created {len(extension_map)} category folders.")

if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    organize_files(target_dir)