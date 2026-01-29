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
import os
import shutil
import tempfile
from pathlib import Path

def clean_temporary_files(directory, extensions=None, days_old=7):
    """
    Remove temporary files from a directory based on extension and age.
    
    Args:
        directory (str): Path to directory to clean
        extensions (list): List of file extensions to target (e.g., ['.tmp', '.temp'])
        days_old (int): Only remove files older than this many days
    """
    if extensions is None:
        extensions = ['.tmp', '.temp', '.log', '.cache']
    
    target_dir = Path(directory)
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    if not target_dir.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    
    current_time = os.path.getmtime(target_dir)
    removed_count = 0
    total_size = 0
    
    for item in target_dir.rglob('*'):
        if item.is_file():
            file_ext = item.suffix.lower()
            
            if file_ext in extensions:
                file_age = current_time - os.path.getmtime(item)
                age_in_days = file_age / (60 * 60 * 24)
                
                if age_in_days > days_old:
                    try:
                        file_size = item.stat().st_size
                        item.unlink()
                        removed_count += 1
                        total_size += file_size
                        print(f"Removed: {item.name} ({file_size} bytes)")
                    except (PermissionError, OSError) as e:
                        print(f"Failed to remove {item.name}: {e}")
    
    print(f"\nCleaning complete:")
    print(f"  Files removed: {removed_count}")
    print(f"  Total space freed: {total_size} bytes")
    return removed_count, total_size

def create_sample_temporary_files(directory, count=5):
    """
    Create sample temporary files for testing purposes.
    
    Args:
        directory (str): Directory to create test files in
        count (int): Number of files to create
    """
    test_dir = Path(directory)
    test_dir.mkdir(exist_ok=True)
    
    extensions = ['.tmp', '.temp', '.log']
    
    for i in range(count):
        ext = extensions[i % len(extensions)]
        temp_file = test_dir / f"test_file_{i}{ext}"
        temp_file.write_text(f"This is a temporary test file #{i}\n" * 100)
        
        if i % 2 == 0:
            old_time = os.path.getmtime(temp_file) - (10 * 24 * 60 * 60)
            os.utime(temp_file, (old_time, old_time))
    
    print(f"Created {count} sample temporary files in {directory}")

if __name__ == "__main__":
    # Example usage
    test_dir = tempfile.mkdtemp(prefix="clean_test_")
    print(f"Test directory: {test_dir}")
    
    try:
        create_sample_temporary_files(test_dir, 8)
        print("\n" + "="*50)
        removed, size = clean_temporary_files(test_dir, days_old=5)
    finally:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
            print(f"\nCleaned up test directory: {test_dir}")