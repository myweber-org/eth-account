
import os
import sys

def rename_files_with_sequence(directory, prefix="file", start_number=1):
    """
    Rename all files in the specified directory with sequential numbering.
    Keeps original file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return False

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()

    counter = start_number
    renamed_count = 0

    for filename in files:
        name, extension = os.path.splitext(filename)
        new_name = f"{prefix}_{counter:03d}{extension}"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)

        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
            counter += 1
            renamed_count += 1
        except OSError as e:
            print(f"Failed to rename {filename}: {e}")

    print(f"Renaming complete. {renamed_count} files renamed.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [start_number]")
        sys.exit(1)

    dir_path = sys.argv[1]
    prefix_arg = sys.argv[2] if len(sys.argv) > 2 else "file"
    start_num = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    rename_files_with_sequence(dir_path, prefix_arg, start_num)