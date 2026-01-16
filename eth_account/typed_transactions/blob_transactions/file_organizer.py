
import os
import shutil

def organize_files(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if os.path.isfile(file_path):
            _, file_extension = os.path.splitext(filename)
            extension = file_extension.lower()

            if extension:
                folder_name = extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"

            target_folder = os.path.join(directory_path, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            target_path = os.path.join(target_folder, filename)
            shutil.move(file_path, target_path)
            print(f"Moved '{filename}' to '{folder_name}/'")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into
    subfolders based on their file extensions.
    """
    # Define file type categories and their associated extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.json', '.xml'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    }
    
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    # Create category folders if they don't exist
    for category in categories.keys():
        category_path = os.path.join(directory, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
    
    # Track moved files and errors
    moved_files = []
    errors = []
    
    # Iterate over all items in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # Skip directories and hidden files
        if os.path.isdir(item_path) or item.startswith('.'):
            continue
        
        # Get file extension
        file_extension = Path(item).suffix.lower()
        
        # Find the appropriate category
        target_category = 'Other'  # Default category
        for category, extensions in categories.items():
            if file_extension in extensions:
                target_category = category
                break
        
        # Create target directory if it doesn't exist (for 'Other' category)
        target_dir = os.path.join(directory, target_category)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Move the file
        try:
            target_path = os.path.join(target_dir, item)
            # Handle filename conflicts
            if os.path.exists(target_path):
                base_name = Path(item).stem
                counter = 1
                while os.path.exists(target_path):
                    new_name = f"{base_name}_{counter}{file_extension}"
                    target_path = os.path.join(target_dir, new_name)
                    counter += 1
            
            shutil.move(item_path, target_path)
            moved_files.append((item, target_category))
        except Exception as e:
            errors.append((item, str(e)))
    
    # Print summary
    if moved_files:
        print(f"Successfully organized {len(moved_files)} files:")
        for file_name, category in moved_files:
            print(f"  {file_name} -> {category}/")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for file_name, error_msg in errors:
            print(f"  {file_name}: {error_msg}")
    
    return len(moved_files), len(errors)

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_dir = os.getcwd()
    print(f"Organizing files in: {current_dir}")
    success, errors = organize_files(current_dir)
    print(f"\nOperation complete. Moved: {success}, Errors: {errors}")