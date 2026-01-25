
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