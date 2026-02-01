
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