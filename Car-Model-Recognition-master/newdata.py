import os
import shutil

def check_and_delete_folders(root_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            num_images = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
            if num_images < 100:
                print(f"Deleting folder: {folder_name}")
                shutil.rmtree(folder_path)
            else:
                print(f"Folder {folder_name} contains {num_images} images, not deleting.")
        else:
            print(f"{folder_name} is not a folder.")

# Replace 'root_folder' with the path to your root folder containing subfolders
root_folder = 'G:\database'
check_and_delete_folders(root_folder)
