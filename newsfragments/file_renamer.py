
import os
import sys
from pathlib import Path

def rename_files_sequentially(directory, prefix="file", dry_run=True):
    """
    Rename files in the given directory with sequential numbers
    based on their modification time.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False
    
    try:
        files = []
        for item in Path(directory).iterdir():
            if item.is_file():
                stat = item.stat()
                files.append((item, stat.st_mtime))
        
        if not files:
            print("No files found in directory.")
            return True
        
        files.sort(key=lambda x: x[1])
        
        print(f"Found {len(files)} files to rename.")
        print("Preview of changes:" if dry_run else "Renaming files:")
        
        for index, (file_path, _) in enumerate(files, start=1):
            extension = file_path.suffix
            new_name = f"{prefix}_{index:03d}{extension}"
            new_path = file_path.parent / new_name
            
            if dry_run:
                print(f"  {file_path.name} -> {new_name}")
            else:
                try:
                    file_path.rename(new_path)
                    print(f"  Renamed: {file_path.name} -> {new_name}")
                except Exception as e:
                    print(f"  Error renaming {file_path.name}: {e}")
        
        if dry_run:
            print("\nThis was a dry run. To actually rename files, run with --apply flag.")
        
        return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [--prefix PREFIX] [--apply]")
        print("Options:")
        print("  --prefix TEXT    Prefix for renamed files (default: 'file')")
        print("  --apply          Actually rename files (default is dry run)")
        sys.exit(1)
    
    directory = sys.argv[1]
    prefix = "file"
    dry_run = True
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--prefix" and i + 1 < len(sys.argv):
            prefix = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--apply":
            dry_run = False
            i += 1
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            sys.exit(1)
    
    success = rename_files_sequentially(directory, prefix, dry_run)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()