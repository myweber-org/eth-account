import os
import glob
from pathlib import Path

def rename_files_sequential(directory, prefix="file", extension=".txt"):
    files = sorted(Path(directory).iterdir(), key=os.path.getctime)
    count = 1
    
    for file_path in files:
        if file_path.is_file():
            new_name = f"{prefix}_{count:03d}{extension}"
            new_path = file_path.parent / new_name
            file_path.rename(new_path)
            print(f"Renamed: {file_path.name} -> {new_name}")
            count += 1

if __name__ == "__main__":
    target_dir = "./documents"
    if Path(target_dir).exists():
        rename_files_sequential(target_dir, "document", ".pdf")
    else:
        print(f"Directory '{target_dir}' not found.")
import os
import re
import argparse

def rename_files(directory, pattern, replacement):
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
        return

    renamed_count = 0
    for filename in files:
        old_path = os.path.join(directory, filename)
        if not os.path.isfile(old_path):
            continue

        new_filename = re.sub(pattern, replacement, filename)
        if new_filename != filename:
            new_path = os.path.join(directory, new_filename)
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: '{filename}' -> '{new_filename}'")
                renamed_count += 1
            except OSError as e:
                print(f"Error renaming '{filename}': {e}")

    print(f"Total files renamed: {renamed_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files in a directory using regex pattern.")
    parser.add_argument("directory", help="Path to the directory containing files to rename.")
    parser.add_argument("pattern", help="Regex pattern to match in filenames.")
    parser.add_argument("replacement", help="Replacement string for matched pattern.")
    args = parser.parse_args()

    rename_files(args.directory, args.pattern, args.replacement)