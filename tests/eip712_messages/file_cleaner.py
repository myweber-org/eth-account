
import os
import time
from pathlib import Path

def clean_old_files(directory, days=7):
    """
    Remove files older than specified days from given directory.
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    removed_count = 0
    
    for item in Path(directory).iterdir():
        if item.is_file():
            if item.stat().st_mtime < cutoff_time:
                try:
                    item.unlink()
                    print(f"Removed: {item}")
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {item}: {e}")
    
    print(f"Cleanup completed. Removed {removed_count} file(s).")

if __name__ == "__main__":
    target_dir = "/tmp/test_files"
    clean_old_files(target_dir, days=7)