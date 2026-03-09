
import os
import sys

def rename_files(directory, prefix="file"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            extension = os.path.splitext(filename)[1]
            new_name = f"{prefix}_{index:03d}{extension}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
            
        print(f"Successfully renamed {len(files)} files.")
        
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix]")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    name_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    
    rename_files(target_dir, name_prefix)import os
import sys

def rename_files_with_sequence(directory, prefix="file"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            file_extension = os.path.splitext(filename)[1]
            new_name = f"{prefix}_{index:03d}{file_extension}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
        
        print(f"Successfully renamed {len(files)} files.")
        return True
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix]")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    name_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    
    if not os.path.isdir(target_directory):
        print(f"Error: '{target_directory}' is not a valid directory.")
        sys.exit(1)
    
    rename_files_with_sequence(target_directory, name_prefix)
import os
import re
import sys
from pathlib import Path

def rename_files(directory, pattern, replacement, dry_run=True):
    """
    Rename files in a directory matching a regex pattern.
    
    Args:
        directory: Path to directory containing files
        pattern: Regex pattern to match in filenames
        replacement: String to replace matched pattern
        dry_run: If True, only show what would be changed
    """
    dir_path = Path(directory)
    
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a directory")
        return False
    
    renamed_count = 0
    
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            original_name = file_path.name
            new_name = re.sub(pattern, replacement, original_name)
            
            if new_name != original_name:
                new_path = file_path.parent / new_name
                
                if dry_run:
                    print(f"Would rename: '{original_name}' -> '{new_name}'")
                else:
                    try:
                        file_path.rename(new_path)
                        print(f"Renamed: '{original_name}' -> '{new_name}'")
                    except Exception as e:
                        print(f"Error renaming '{original_name}': {e}")
                        continue
                
                renamed_count += 1
    
    print(f"\nTotal files that would be renamed: {renamed_count}")
    if dry_run:
        print("This was a dry run. No files were actually renamed.")
        print("Run with --apply to actually rename files.")
    
    return True

def main():
    if len(sys.argv) < 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement> [--apply]")
        print("Example: python file_renamer.py ./photos 'IMG_\\d+' 'vacation_'")
        return
    
    directory = sys.argv[1]
    pattern = sys.argv[2]
    replacement = sys.argv[3]
    dry_run = '--apply' not in sys.argv
    
    try:
        rename_files(directory, pattern, replacement, dry_run)
    except re.error as e:
        print(f"Invalid regex pattern: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
import os
import sys

def rename_files_with_sequence(directory, prefix="file", extension=".txt"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            old_path = os.path.join(directory, filename)
            new_name = f"{prefix}_{index:03d}{extension}"
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
            
        print(f"Successfully renamed {len(files)} files.")
        return True
        
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return False
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
        extension = sys.argv[3] if len(sys.argv) > 3 else ".txt"
        
        rename_files_with_sequence(target_dir, prefix, extension)
    else:
        print("Usage: python file_renamer.py <directory> [prefix] [extension]")
        print("Example: python file_renamer.py ./my_files document .pdf")