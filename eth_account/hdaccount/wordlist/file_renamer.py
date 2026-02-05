
import os
import sys

def batch_rename_files(directory, prefix, extension):
    """
    Rename all files in the specified directory with sequential numbering.
    
    Args:
        directory: Path to the directory containing files to rename
        prefix: Prefix for the new filenames
        extension: File extension (without dot) for the new filenames
    """
    try:
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return False
        
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        if not files:
            print("No files found in the directory.")
            return False
        
        renamed_count = 0
        for index, filename in enumerate(files, start=1):
            old_path = os.path.join(directory, filename)
            new_filename = f"{prefix}_{index:03d}.{extension}"
            new_path = os.path.join(directory, new_filename)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            except OSError as e:
                print(f"Failed to rename {filename}: {e}")
        
        print(f"\nSuccessfully renamed {renamed_count} out of {len(files)} files.")
        return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <prefix> <extension>")
        print("Example: python file_renamer.py ./photos vacation jpg")
        sys.exit(1)
    
    directory = sys.argv[1]
    prefix = sys.argv[2]
    extension = sys.argv[3]
    
    batch_rename_files(directory, prefix, extension)