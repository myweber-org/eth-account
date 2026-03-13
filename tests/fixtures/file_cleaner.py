
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

class TempFileCleaner:
    def __init__(self, target_dir: Optional[str] = None):
        self.target_dir = Path(target_dir) if target_dir else Path(tempfile.gettempdir())
        self.removed_files = []
        self.removed_dirs = []

    def scan_temp_files(self, patterns: List[str] = None) -> List[Path]:
        if patterns is None:
            patterns = ['*.tmp', 'temp_*', '~*', '*.cache']
        
        found_files = []
        for pattern in patterns:
            found_files.extend(self.target_dir.glob(pattern))
        
        return sorted(set(found_files))

    def cleanup_files(self, max_age_days: int = 7, dry_run: bool = False) -> dict:
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        stats = {
            'files_removed': 0,
            'dirs_removed': 0,
            'total_size': 0,
            'errors': []
        }
        
        for item in self.target_dir.iterdir():
            try:
                if item.is_file():
                    file_age = datetime.fromtimestamp(item.stat().st_mtime)
                    if file_age < cutoff_time:
                        if not dry_run:
                            file_size = item.stat().st_size
                            item.unlink()
                            self.removed_files.append(item)
                            stats['files_removed'] += 1
                            stats['total_size'] += file_size
                        else:
                            stats['files_removed'] += 1
                
                elif item.is_dir() and item.name.startswith('tmp'):
                    dir_age = datetime.fromtimestamp(item.stat().st_mtime)
                    if dir_age < cutoff_time:
                        if not dry_run:
                            shutil.rmtree(item)
                            self.removed_dirs.append(item)
                            stats['dirs_removed'] += 1
                        else:
                            stats['dirs_removed'] += 1
                            
            except (OSError, PermissionError) as e:
                stats['errors'].append(str(e))
        
        return stats

    def get_summary(self) -> str:
        total_files = len(self.removed_files)
        total_dirs = len(self.removed_dirs)
        
        if total_files == 0 and total_dirs == 0:
            return "No items were cleaned up."
        
        summary = f"Cleanup completed:\n"
        summary += f"- Files removed: {total_files}\n"
        summary += f"- Directories removed: {total_dirs}\n"
        
        if self.removed_files:
            summary += "\nRemoved files:\n"
            for f in self.removed_files[:10]:
                summary += f"  {f.name}\n"
            if len(self.removed_files) > 10:
                summary += f"  ... and {len(self.removed_files) - 10} more\n"
        
        return summary

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean temporary files from a directory')
    parser.add_argument('--dir', '-d', help='Target directory to clean')
    parser.add_argument('--days', '-t', type=int, default=7, 
                       help='Remove files older than N days (default: 7)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be removed without actually deleting')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    cleaner = TempFileCleaner(args.dir)
    
    if args.verbose:
        print(f"Scanning directory: {cleaner.target_dir}")
        temp_files = cleaner.scan_temp_files()
        print(f"Found {len(temp_files)} temporary file patterns")
    
    stats = cleaner.cleanup_files(max_age_days=args.days, dry_run=args.dry_run)
    
    if args.dry_run:
        print("DRY RUN - No files will be deleted")
    
    print(cleaner.get_summary())
    
    if args.verbose and stats['errors']:
        print("\nErrors encountered:")
        for error in stats['errors']:
            print(f"  - {error}")

if __name__ == '__main__':
    main()
import os
import re
import unicodedata

def clean_filename(filename):
    """
    Normalize and clean a filename by removing or replacing invalid characters,
    normalizing unicode, and trimming extra spaces.
    """
    # Normalize unicode characters (e.g., convert accented characters to ASCII)
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    # Remove any characters that are not alphanumeric, underscore, dash, or dot
    filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '', filename)
    
    # Remove leading/trailing underscores, dashes, or dots
    filename = filename.strip('_-.')
    
    # Convert to lowercase (optional, can be removed if case should be preserved)
    filename = filename.lower()
    
    # Limit length (optional, e.g., max 255 characters typical for filesystems)
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        # Truncate name part, keep extension
        filename = name[:max_length - len(ext)] + ext
    
    return filename

def clean_filenames_in_directory(directory_path):
    """
    Iterate through all files in the given directory and rename them
    using the clean_filename function.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return
    
    for filename in os.listdir(directory_path):
        old_path = os.path.join(directory_path, filename)
        
        # Skip directories
        if os.path.isdir(old_path):
            continue
        
        new_filename = clean_filename(filename)
        new_path = os.path.join(directory_path, new_filename)
        
        # Only rename if the filename actually changed
        if old_path != new_path:
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
            except OSError as e:
                print(f"Error renaming {filename}: {e}")

if __name__ == "__main__":
    # Example usage: clean filenames in the current directory
    import sys
    
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    else:
        target_directory = "."
    
    clean_filenames_in_directory(target_directory)