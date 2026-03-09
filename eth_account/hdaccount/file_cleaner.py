
import os
import time
from pathlib import Path

def clean_old_files(directory, days=7):
    """
    Remove files in the specified directory that are older than the given number of days.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return
    
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    removed_count = 0
    
    for item in Path(directory).iterdir():
        if item.is_file():
            file_mtime = item.stat().st_mtime
            if file_mtime < cutoff_time:
                try:
                    item.unlink()
                    print(f"Removed: {item.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"Failed to remove {item.name}: {e}")
    
    print(f"Cleanup completed. {removed_count} files removed.")

if __name__ == "__main__":
    target_dir = "/tmp/my_temp_files"
    clean_old_files(target_dir, days=7)
import os
import sys

def clean_directory(directory_path, remove_empty_dirs=True, dry_run=False):
    """
    Remove temporary files and optionally empty directories.
    
    Args:
        directory_path: Path to directory to clean
        remove_empty_dirs: Whether to remove empty directories
        dry_run: If True, only show what would be removed without actually removing
    """
    if not os.path.exists(directory_path):
        print(f"Directory does not exist: {directory_path}")
        return
    
    if not os.path.isdir(directory_path):
        print(f"Path is not a directory: {directory_path}")
        return
    
    temp_extensions = ['.tmp', '.temp', '.bak', '.swp', '.~']
    removed_files = []
    removed_dirs = []
    
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in temp_extensions:
                if dry_run:
                    print(f"[DRY RUN] Would remove file: {file_path}")
                else:
                    try:
                        os.remove(file_path)
                        removed_files.append(file_path)
                        print(f"Removed file: {file_path}")
                    except OSError as e:
                        print(f"Error removing file {file_path}: {e}")
        
        if remove_empty_dirs:
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):
                        if dry_run:
                            print(f"[DRY RUN] Would remove empty directory: {dir_path}")
                        else:
                            os.rmdir(dir_path)
                            removed_dirs.append(dir_path)
                            print(f"Removed empty directory: {dir_path}")
                except OSError as e:
                    print(f"Error checking directory {dir_path}: {e}")
    
    summary = f"\nSummary:\n"
    summary += f"Files removed: {len(removed_files)}\n"
    summary += f"Directories removed: {len(removed_dirs)}\n"
    
    if dry_run:
        summary = summary.replace("removed", "would be removed")
    
    print(summary)
    
    return {
        'files_removed': removed_files,
        'dirs_removed': removed_dirs,
        'total_removed': len(removed_files) + len(removed_dirs)
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <directory_path> [--dry-run] [--keep-dirs]")
        print("Options:")
        print("  --dry-run    Show what would be removed without actually removing")
        print("  --keep-dirs  Do not remove empty directories")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    dry_run = '--dry-run' in sys.argv
    keep_dirs = '--keep-dirs' in sys.argv
    
    clean_directory(
        directory_path=directory_path,
        remove_empty_dirs=not keep_dirs,
        dry_run=dry_run
    )

if __name__ == "__main__":
    main()
import os
import sys
import shutil
import argparse

def clean_directory(directory, extensions, dry_run=False):
    """
    Remove files with specified extensions from the given directory.
    If dry_run is True, only print the files that would be removed.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False

    removed_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                if dry_run:
                    print(f"[Dry Run] Would remove: {file_path}")
                else:
                    try:
                        os.remove(file_path)
                        print(f"Removed: {file_path}")
                        removed_count += 1
                    except OSError as e:
                        print(f"Error removing {file_path}: {e}")
    return removed_count

def main():
    parser = argparse.ArgumentParser(description="Clean temporary files from a directory.")
    parser.add_argument("directory", help="Directory to clean")
    parser.add_argument("-e", "--extensions", nargs="+", default=[".tmp", ".log", ".bak"],
                        help="File extensions to remove (default: .tmp .log .bak)")
    parser.add_argument("-d", "--dry-run", action="store_true",
                        help="Perform a dry run without deleting files")

    args = parser.parse_args()

    print(f"Cleaning directory: {args.directory}")
    print(f"Extensions to remove: {args.extensions}")
    if args.dry_run:
        print("Running in dry-run mode. No files will be deleted.")

    removed = clean_directory(args.directory, args.extensions, args.dry_run)

    if args.dry_run:
        print(f"Dry run complete. {removed} files would be removed.")
    else:
        print(f"Cleanup complete. {removed} files removed.")

if __name__ == "__main__":
    main()import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

class TempFileCleaner:
    """A utility class to clean temporary files and directories."""

    def __init__(self, target_dir: Optional[str] = None):
        """
        Initialize the cleaner with a target directory.
        If no directory is provided, uses the system's temp directory.
        """
        self.target_dir = Path(target_dir) if target_dir else Path(tempfile.gettempdir())
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.target_dir}")

    def list_temp_files(self, pattern: str = "*") -> List[Path]:
        """List files in the target directory matching a pattern."""
        return list(self.target_dir.glob(pattern))

    def clean_files(self, pattern: str = "*", dry_run: bool = True) -> List[Path]:
        """
        Remove files matching the pattern.
        If dry_run is True, only list files without deleting.
        Returns list of files that would be/were removed.
        """
        files_to_clean = self.list_temp_files(pattern)
        removed = []

        for file_path in files_to_clean:
            try:
                if dry_run:
                    removed.append(file_path)
                else:
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                    removed.append(file_path)
            except (OSError, PermissionError) as e:
                print(f"Error processing {file_path}: {e}")

        return removed

    def clean_old_files(self, days_old: int = 7, dry_run: bool = True) -> List[Path]:
        """Remove files older than specified number of days."""
        import time
        current_time = time.time()
        cutoff = current_time - (days_old * 86400)
        old_files = []

        for file_path in self.list_temp_files():
            try:
                if file_path.stat().st_mtime < cutoff:
                    if dry_run:
                        old_files.append(file_path)
                    else:
                        if file_path.is_file():
                            file_path.unlink()
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                        old_files.append(file_path)
            except (OSError, PermissionError) as e:
                print(f"Error processing {file_path}: {e}")

        return old_files

def main():
    """Example usage of the TempFileCleaner."""
    cleaner = TempFileCleaner()
    
    print("Listing temporary files:")
    temp_files = cleaner.list_temp_files()
    for f in temp_files[:5]:  # Show first 5
        print(f"  {f}")
    
    print("\nDry run - files that would be cleaned:")
    to_clean = cleaner.clean_files("*.tmp", dry_run=True)
    for f in to_clean[:5]:
        print(f"  {f}")
    
    print(f"\nTotal files in temp dir: {len(temp_files)}")
    print(f"Files to clean with pattern '*.tmp': {len(to_clean)}")

if __name__ == "__main__":
    main()