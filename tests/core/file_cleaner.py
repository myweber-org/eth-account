import os
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_old_files(directory_path, days_old=7):
    """
    Remove files older than specified days from given directory.
    """
    if not os.path.exists(directory_path):
        logger.error(f"Directory does not exist: {directory_path}")
        return

    cutoff_time = time.time() - (days_old * 86400)
    deleted_count = 0
    error_count = 0

    for item in Path(directory_path).iterdir():
        try:
            if item.is_file():
                if item.stat().st_mtime < cutoff_time:
                    item.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted: {item.name}")
        except Exception as e:
            error_count += 1
            logger.error(f"Failed to delete {item.name}: {e}")

    logger.info(f"Cleanup completed. Deleted: {deleted_count}, Errors: {error_count}")

if __name__ == "__main__":
    target_dir = "/tmp/my_app_cache"
    clean_old_files(target_dir, days_old=7)