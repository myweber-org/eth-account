
import os
import shutil

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1].lower() if '.' in filename else 'no_extension'
            
            folder_name = get_folder_name(file_extension)
            folder_path = os.path.join(directory, folder_name)
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            destination_path = os.path.join(folder_path, filename)
            shutil.move(file_path, destination_path)
            print(f"Moved {filename} to {folder_name}/")

def get_folder_name(extension):
    extension_folders = {
        'txt': 'TextFiles',
        'pdf': 'Documents',
        'jpg': 'Images',
        'png': 'Images',
        'mp3': 'Audio',
        'mp4': 'Videos',
        'py': 'PythonScripts',
        'zip': 'Archives'
    }
    
    return extension_folders.get(extension, 'OtherFiles')

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)