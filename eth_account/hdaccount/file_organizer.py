
import os
import shutil

def organize_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            extension = extension.lower()
            
            if extension:
                folder_name = extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            target_folder = os.path.join(directory, folder_name)
            
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            
            target_path = os.path.join(target_folder, filename)
            
            if not os.path.exists(target_path):
                shutil.move(file_path, target_path)
                print(f"Moved: {filename} -> {folder_name}/")
            else:
                print(f"Skipped: {filename} already exists in {folder_name}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)