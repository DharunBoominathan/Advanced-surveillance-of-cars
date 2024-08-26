import os
import shutil

def combine_folders(source_folder, destination_folder):
    # Iterate over all files and folders in the source folder
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)
        
        # If it's a folder, recursively call combine_folders
        if os.path.isdir(item_path):
            combine_folders(item_path, destination_folder)
        else:
            # Get the model name from the file name (split by underscore and join the first two parts)
            model_name = '_'.join(item.split('_')[:2])
            # Create destination folder if it doesn't exist
            model_folder = os.path.join(destination_folder, model_name)
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            # Copy the file to the model folder
            shutil.copy(item_path, os.path.join(model_folder, item))

# Source folder containing all the subfolders
source_folder = 'G:\databasesmall\database'
# Destination folder where all files will be combined
destination_folder = 'G:\databasesmall\\new'

combine_folders(source_folder, destination_folder)
