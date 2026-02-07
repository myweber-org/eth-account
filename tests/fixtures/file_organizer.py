
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            target_path = os.path.join(target_folder, item)
            
            try:
                shutil.move(item_path, target_path)
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization completed.")
import os
import shutil

def organize_files(directory):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext:
                target_folder = os.path.join(directory, file_ext[1:] + "_files")
            else:
                target_folder = os.path.join(directory, "no_extension_files")

            os.makedirs(target_folder, exist_ok=True)

            try:
                shutil.move(file_path, os.path.join(target_folder, filename))
                print(f"Moved: {filename} -> {target_folder}")
            except Exception as e:
                print(f"Failed to move {filename}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path

class FileOrganizer:
    def __init__(self, source_dir, target_base_dir):
        self.source_dir = Path(source_dir)
        self.target_base_dir = Path(target_base_dir)
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('file_organizer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def categorize_file(self, file_path):
        """Categorize files based on extension and size"""
        if not file_path.is_file():
            return None
            
        extension = file_path.suffix.lower()
        file_size = file_path.stat().st_size
        
        categories = {
            '.jpg': 'images', '.jpeg': 'images', '.png': 'images', '.gif': 'images',
            '.pdf': 'documents', '.doc': 'documents', '.docx': 'documents', '.txt': 'documents',
            '.mp4': 'videos', '.avi': 'videos', '.mov': 'videos',
            '.mp3': 'audio', '.wav': 'audio', '.flac': 'audio',
            '.py': 'scripts', '.js': 'scripts', '.html': 'scripts',
            '.zip': 'archives', '.rar': 'archives', '.tar': 'archives'
        }
        
        category = categories.get(extension, 'others')
        
        if file_size > 100 * 1024 * 1024:
            category = f'large_files/{category}'
        elif file_size < 1024:
            category = f'tiny_files/{category}'
            
        return category
    
    def organize_files(self):
        """Main organization method"""
        if not self.source_dir.exists():
            self.logger.error(f"Source directory does not exist: {self.source_dir}")
            return False
            
        self.target_base_dir.mkdir(parents=True, exist_ok=True)
        processed_count = 0
        error_count = 0
        
        for item in self.source_dir.iterdir():
            try:
                if item.is_file():
                    category = self.categorize_file(item)
                    if category:
                        target_dir = self.target_base_dir / category
                        target_dir.mkdir(parents=True, exist_ok=True)
                        
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        new_name = f"{timestamp}_{item.name}"
                        target_path = target_dir / new_name
                        
                        shutil.move(str(item), str(target_path))
                        self.logger.info(f"Moved: {item.name} -> {category}/{new_name}")
                        processed_count += 1
                        
            except Exception as e:
                self.logger.error(f"Error processing {item.name}: {str(e)}")
                error_count += 1
                continue
                
        self.logger.info(f"Organization complete. Processed: {processed_count}, Errors: {error_count}")
        return processed_count > 0
    
    def generate_report(self):
        """Generate organization report"""
        report_file = self.target_base_dir / 'organization_report.txt'
        with open(report_file, 'w') as f:
            f.write("File Organization Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Source: {self.source_dir}\n")
            f.write(f"Target: {self.target_base_dir}\n\n")
            
            total_files = 0
            for category_dir in self.target_base_dir.rglob('*'):
                if category_dir.is_dir():
                    files = list(category_dir.glob('*'))
                    if files:
                        f.write(f"{category_dir.relative_to(self.target_base_dir)}: {len(files)} files\n")
                        total_files += len(files)
            
            f.write(f"\nTotal files organized: {total_files}\n")
        
        self.logger.info(f"Report generated: {report_file}")
        return report_file

def main():
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python file_organizer.py <source_directory> <target_directory>")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    
    organizer = FileOrganizer(source_dir, target_dir)
    
    if organizer.organize_files():
        organizer.generate_report()
        print("File organization completed successfully!")
    else:
        print("No files were organized. Check the logs for details.")

if __name__ == "__main__":
    main()
import os
import shutil
from pathlib import Path

def organize_files(source_dir, organize_by='extension'):
    """
    Organize files in the source directory into subfolders based on criteria.
    """
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)

        if os.path.isfile(item_path):
            if organize_by == 'extension':
                file_ext = Path(item).suffix.lower()
                if file_ext:
                    folder_name = file_ext[1:] + '_files'
                else:
                    folder_name = 'no_extension_files'
            else:
                folder_name = 'other_files'

            target_dir = os.path.join(source_dir, folder_name)
            os.makedirs(target_dir, exist_ok=True)

            try:
                shutil.move(item_path, os.path.join(target_dir, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    if target_directory:
        organize_files(target_directory)
    else:
        print("No directory provided. Exiting.")
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """Organize files in the given directory by their extensions."""
    base_path = Path(directory)
    
    # Define categories and their associated file extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md"],
        "Archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c"],
        "Audio": [".mp3", ".wav", ".flac", ".aac"],
        "Video": [".mp4", ".avi", ".mov", ".mkv", ".flv"]
    }
    
    # Create category folders if they don't exist
    for category in categories:
        (base_path / category).mkdir(exist_ok=True)
    
    # Track moved files for reporting
    moved_files = []
    
    # Iterate through files in the directory
    for item in base_path.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            moved = False
            
            # Find the appropriate category for the file
            for category, extensions in categories.items():
                if file_ext in extensions:
                    target_dir = base_path / category
                    try:
                        shutil.move(str(item), str(target_dir / item.name))
                        moved_files.append((item.name, category))
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")
            
            # If file doesn't match any category, move to "Other"
            if not moved:
                other_dir = base_path / "Other"
                other_dir.mkdir(exist_ok=True)
                try:
                    shutil.move(str(item), str(other_dir / item.name))
                    moved_files.append((item.name, "Other"))
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")
    
    # Print summary
    if moved_files:
        print(f"Organized {len(moved_files)} file(s):")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    else:
        print("No files to organize.")

if __name__ == "__main__":
    organize_files()