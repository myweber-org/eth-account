
import os
import re
import sys

def rename_files(directory, pattern, replacement):
    """
    Rename files in the specified directory based on a regex pattern.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        pattern (str): Regex pattern to match in filenames.
        replacement (str): Replacement string for matched pattern.
    """
    try:
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return

        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        renamed_count = 0

        for filename in files:
            new_name = re.sub(pattern, replacement, filename)
            if new_name != filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_name)
                
                # Avoid overwriting existing files
                if os.path.exists(new_path):
                    print(f"Warning: '{new_name}' already exists. Skipping '{filename}'.")
                    continue
                
                os.rename(old_path, new_path)
                print(f"Renamed: '{filename}' -> '{new_name}'")
                renamed_count += 1

        print(f"Renaming complete. {renamed_count} file(s) renamed.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    regex_pattern = sys.argv[2]
    replace_with = sys.argv[3]
    
    rename_files(target_dir, regex_pattern, replace_with)