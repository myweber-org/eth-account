
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
    main()