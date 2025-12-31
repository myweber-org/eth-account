
import os
import shutil
import argparse

def clean_directory(directory, extensions_to_remove):
    """
    Remove files with specified extensions from the given directory.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    removed_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions_to_remove):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    removed_files.append(file_path)
                    print(f"Removed: {file_path}")
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")

    return removed_files

def main():
    parser = argparse.ArgumentParser(description="Clean temporary files from a directory.")
    parser.add_argument("directory", help="Directory to clean")
    parser.add_argument("-e", "--extensions", nargs="+", default=[".tmp", ".log", ".bak"],
                        help="File extensions to remove (default: .tmp .log .bak)")

    args = parser.parse_args()

    print(f"Cleaning directory: {args.directory}")
    print(f"Removing files with extensions: {args.extensions}")
    
    removed = clean_directory(args.directory, args.extensions)
    
    if removed:
        print(f"\nCleaning complete. Removed {len(removed)} file(s).")
    else:
        print("\nNo files matching the specified extensions were found.")

if __name__ == "__main__":
    main()