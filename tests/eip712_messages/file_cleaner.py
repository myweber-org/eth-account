
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
import sys
import os

def remove_duplicates(input_file, output_file=None):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return False
    
    if output_file is None:
        output_file = input_file + ".deduped"
    
    seen_lines = set()
    lines_removed = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        unique_lines = []
        for line in lines:
            stripped_line = line.rstrip('\n')
            if stripped_line not in seen_lines:
                seen_lines.add(stripped_line)
                unique_lines.append(line)
            else:
                lines_removed += 1
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)
        
        print(f"Successfully removed {lines_removed} duplicate lines.")
        print(f"Original lines: {len(lines)}")
        print(f"Unique lines: {len(unique_lines)}")
        print(f"Output saved to: {output_file}")
        
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
    
    success = remove_duplicates(input_file, output_file)
    sys.exit(0 if success else 1)