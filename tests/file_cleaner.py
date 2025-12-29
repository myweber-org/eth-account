import os
import shutil
import tempfile
from pathlib import Path

def clean_temp_files(directory: str, extensions: tuple = ('.tmp', '.temp', '.log'), days_old: int = 7):
    """
    Remove temporary files from a directory based on extension and age.
    
    Args:
        directory: Path to the directory to clean.
        extensions: Tuple of file extensions to consider as temporary.
        days_old: Remove files older than this many days.
    """
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Invalid directory: {directory}")

    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)

    for item in dir_path.rglob('*'):
        if item.is_file() and item.suffix.lower() in extensions:
            file_age = item.stat().st_mtime
            if file_age < cutoff_time:
                try:
                    item.unlink()
                    print(f"Removed: {item}")
                except OSError as e:
                    print(f"Error removing {item}: {e}")
        elif item.is_dir() and item.name == '__pycache__':
            try:
                shutil.rmtree(item)
                print(f"Removed __pycache__: {item}")
            except OSError as e:
                print(f"Error removing {item}: {e}")

def create_test_environment():
    """Create a test directory with temporary files for demonstration."""
    test_dir = tempfile.mkdtemp(prefix='clean_test_')
    print(f"Created test directory: {test_dir}")
    
    # Create some test files
    extensions = ['.tmp', '.temp', '.log', '.txt']
    for i in range(10):
        ext = extensions[i % len(extensions)]
        temp_file = Path(test_dir) / f"test_file_{i}{ext}"
        temp_file.write_text(f"Test content {i}")
        
        # Make some files older
        if i % 3 == 0:
            old_time = time.time() - (10 * 24 * 60 * 60)
            os.utime(temp_file, (old_time, old_time))
    
    # Create a __pycache__ directory
    cache_dir = Path(test_dir) / '__pycache__'
    cache_dir.mkdir()
    (cache_dir / 'test.cpython-39.pyc').write_text('bytecode')
    
    return test_dir

if __name__ == '__main__':
    import time
    
    # Create and clean a test environment
    test_env = create_test_environment()
    print(f"\nCleaning directory: {test_env}")
    clean_temp_files(test_env, days_old=5)
    
    # Clean up test directory
    shutil.rmtree(test_env)
    print(f"\nRemoved test directory: {test_env}")