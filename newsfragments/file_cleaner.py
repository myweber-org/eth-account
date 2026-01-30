
import os
import shutil
import tempfile
from pathlib import Path

def clean_temporary_files(directory_path, extensions=None, days_old=7):
    """
    Remove temporary files from a specified directory.
    
    Args:
        directory_path (str or Path): Path to the directory to clean.
        extensions (list, optional): List of file extensions to consider as temporary.
                                     Defaults to common temporary extensions.
        days_old (int, optional): Remove files older than this many days. Defaults to 7.
    
    Returns:
        dict: Summary of cleaned files with counts and total size freed.
    """
    if extensions is None:
        extensions = ['.tmp', '.temp', '.log', '.cache', '.bak', '.swp']
    
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    summary = {
        'files_removed': 0,
        'bytes_freed': 0,
        'errors': []
    }
    
    current_time = os.path.getctime(directory)
    cutoff_time = current_time - (days_old * 24 * 60 * 60)
    
    for item in directory.rglob('*'):
        try:
            if item.is_file():
                file_extension = item.suffix.lower()
                
                if file_extension in extensions or item.stat().st_ctime < cutoff_time:
                    file_size = item.stat().st_size
                    item.unlink()
                    
                    summary['files_removed'] += 1
                    summary['bytes_freed'] += file_size
                    
        except (OSError, PermissionError) as e:
            summary['errors'].append(str(e))
            continue
    
    summary['bytes_freed_mb'] = summary['bytes_freed'] / (1024 * 1024)
    
    return summary

def create_test_environment():
    """Create a test directory with temporary files for demonstration."""
    test_dir = tempfile.mkdtemp(prefix='clean_test_')
    print(f"Created test directory: {test_dir}")
    
    test_files = [
        'document.tmp',
        'backup.bak',
        'cache.cache',
        'error.log',
        'important.txt',
        'config.yaml'
    ]
    
    for filename in test_files:
        file_path = Path(test_dir) / filename
        file_path.touch()
    
    return test_dir

if __name__ == "__main__":
    try:
        test_directory = create_test_environment()
        
        print("Cleaning temporary files...")
        result = clean_temporary_files(test_directory)
        
        print(f"Files removed: {result['files_removed']}")
        print(f"Space freed: {result['bytes_freed_mb']:.2f} MB")
        
        if result['errors']:
            print(f"Encountered {len(result['errors'])} errors during cleanup")
        
        shutil.rmtree(test_directory)
        print("Test cleanup completed successfully")
        
    except Exception as e:
        print(f"An error occurred: {e}")