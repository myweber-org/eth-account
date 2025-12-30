import os
import shutil
import sys

def clean_directory(directory_path, extensions_to_remove=None, dry_run=False):
    """
    Remove temporary files from a directory based on given extensions.
    If extensions_to_remove is None, uses a default list of common temp extensions.
    """
    if extensions_to_remove is None:
        extensions_to_remove = ['.tmp', '.temp', '.bak', '.log', '.cache']
    
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return False
    
    removed_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in extensions_to_remove:
                if dry_run:
                    print(f"[DRY RUN] Would remove: {file_path}")
                else:
                    try:
                        os.remove(file_path)
                        removed_files.append(file_path)
                        print(f"Removed: {file_path}")
                    except OSError as e:
                        print(f"Error removing {file_path}: {e}")
    
    if not dry_run:
        print(f"\nCleaning complete. Removed {len(removed_files)} files.")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <directory_path> [--dry-run]")
        print("Example: python file_cleaner.py /path/to/dir --dry-run")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    dry_run = '--dry-run' in sys.argv
    
    clean_directory(directory_path, dry_run=dry_run)

if __name__ == "__main__":
    main()