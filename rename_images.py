import os
import re

def remove_non_ascii_characters(filename):
    """Remove non-ASCII characters from the filename."""
    return re.sub(r'[^\x00-\x7F]', '', filename)

def rename_images_in_folder(folder_path):
    """Rename images in the specified folder, removing Japanese and Chinese characters from the filenames."""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            new_filename = remove_non_ascii_characters(filename)
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    folder_path = r"D:\PythonProject\objectC model"
    rename_images_in_folder(folder_path)
