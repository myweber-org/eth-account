import os
import time
from pathlib import Path

def remove_old_files(directory, days_old=7):
    """
    Remove files in the specified directory that are older than the given number of days.
    """
    if not os.path.isdir(directory):
        print(f"Directory does not exist: {directory}")
        return
    
    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)
    removed_count = 0
    
    for item in Path(directory).iterdir():
        if item.is_file():
            file_mtime = item.stat().st_mtime
            if file_mtime < cutoff_time:
                try:
                    item.unlink()
                    print(f"Removed: {item}")
                    removed_count += 1
                except OSError as e:
                    print(f"Error removing {item}: {e}")
    
    print(f"Removed {removed_count} file(s) older than {days_old} days.")

if __name__ == "__main__":
    target_dir = "/tmp/my_temp_files"
    remove_old_files(target_dir)