
import os
import shutil
import sys

def clean_directory(directory, extensions_to_remove=None, dry_run=False):
    """
    Remove temporary files from a directory.
    
    Args:
        directory (str): Path to the directory to clean.
        extensions_to_remove (list): List of file extensions to remove.
                                    If None, uses default list.
        dry_run (bool): If True, only print what would be removed without deleting.
    
    Returns:
        dict: Statistics about the cleaning operation.
    """
    if extensions_to_remove is None:
        extensions_to_remove = ['.tmp', '.temp', '.log', '.bak', '~']
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return {'error': 'Directory not found'}
    
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.")
        return {'error': 'Not a directory'}
    
    removed_files = []
    removed_size = 0
    skipped_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            should_remove = False
            for ext in extensions_to_remove:
                if file.endswith(ext):
                    should_remove = True
                    break
            
            if should_remove:
                try:
                    file_size = os.path.getsize(file_path)
                    
                    if dry_run:
                        print(f"[DRY RUN] Would remove: {file_path} ({file_size} bytes)")
                    else:
                        os.remove(file_path)
                        print(f"Removed: {file_path} ({file_size} bytes)")
                    
                    removed_files.append(file_path)
                    removed_size += file_size
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
                    skipped_files.append(file_path)
            else:
                skipped_files.append(file_path)
    
    stats = {
        'removed_files': len(removed_files),
        'removed_size': removed_size,
        'skipped_files': len(skipped_files),
        'dry_run': dry_run
    }
    
    print(f"\nCleaning completed:")
    print(f"  Files removed: {len(removed_files)}")
    print(f"  Total size freed: {removed_size} bytes")
    print(f"  Files skipped: {len(skipped_files)}")
    
    if dry_run:
        print("  Note: This was a dry run. No files were actually deleted.")
    
    return stats

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Clean temporary files from a directory.'
    )
    parser.add_argument(
        'directory',
        help='Directory to clean'
    )
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.tmp', '.temp', '.log', '.bak', '~'],
        help='File extensions to remove (space separated)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually deleting'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    
    args = parser.parse_args()
    
    if not args.verbose:
        import logging
        logging.basicConfig(level=logging.WARNING)
    
    stats = clean_directory(
        args.directory,
        extensions_to_remove=args.extensions,
        dry_run=args.dry_run
    )
    
    if 'error' in stats:
        sys.exit(1)

if __name__ == '__main__':
    main()import sys
import os

def remove_duplicates(input_file, output_file=None):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return False
    
    if output_file is None:
        output_file = input_file + ".deduped"
    
    seen_lines = set()
    unique_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.rstrip('\n')
                if stripped_line not in seen_lines:
                    seen_lines.add(stripped_line)
                    unique_lines.append(line)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)
        
        print(f"Successfully removed duplicates. Output saved to '{output_file}'.")
        print(f"Original lines: {len(seen_lines) + (len(unique_lines) - len(seen_lines))}")
        print(f"Unique lines: {len(unique_lines)}")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)