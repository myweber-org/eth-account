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
import os
import shutil
import tempfile
from pathlib import Path

class TemporaryFileCleaner:
    def __init__(self, target_dir=None):
        self.target_dir = Path(target_dir) if target_dir else Path(tempfile.gettempdir())
        self.deleted_files = []
        self.deleted_dirs = []

    def scan_and_clean(self, extensions=None, days_old=7):
        if extensions is None:
            extensions = ['.tmp', '.temp', '.log', '.cache']
        
        current_time = time.time()
        cutoff_time = current_time - (days_old * 86400)

        for item in self.target_dir.rglob('*'):
            try:
                if item.is_file():
                    if item.suffix.lower() in extensions or item.stat().st_mtime < cutoff_time:
                        self._remove_file(item)
                elif item.is_dir() and not any(item.iterdir()):
                    self._remove_dir(item)
            except (PermissionError, OSError) as e:
                print(f"Could not process {item}: {e}")

    def _remove_file(self, file_path):
        try:
            file_path.unlink()
            self.deleted_files.append(str(file_path))
            print(f"Removed file: {file_path}")
        except Exception as e:
            print(f"Failed to remove {file_path}: {e}")

    def _remove_dir(self, dir_path):
        try:
            dir_path.rmdir()
            self.deleted_dirs.append(str(dir_path))
            print(f"Removed empty directory: {dir_path}")
        except Exception as e:
            print(f"Failed to remove {dir_path}: {e}")

    def get_summary(self):
        return {
            'target_directory': str(self.target_dir),
            'files_deleted': len(self.deleted_files),
            'directories_deleted': len(self.deleted_dirs),
            'total_space_freed': self._calculate_freed_space()
        }

    def _calculate_freed_space(self):
        total_size = 0
        for file_path in self.deleted_files:
            try:
                total_size += Path(file_path).stat().st_size
            except:
                pass
        return total_size

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean temporary files from a directory')
    parser.add_argument('--dir', help='Target directory to clean')
    parser.add_argument('--extensions', nargs='+', default=['.tmp', '.temp'],
                       help='File extensions to remove')
    parser.add_argument('--days', type=int, default=7,
                       help='Remove files older than X days')
    
    args = parser.parse_args()
    
    cleaner = TemporaryFileCleaner(args.dir)
    cleaner.scan_and_clean(args.extensions, args.days)
    
    summary = cleaner.get_summary()
    print("\nCleanup Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    import time
    main()