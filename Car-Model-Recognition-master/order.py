import os

def rename_files(folder_path,name):
    # List all files in the folder
    files = os.listdir(folder_path)
    # Sort the files alphabetically
    files.sort()
    
    # Initialize a counter for renaming
    count = 1
    
    # Iterate over each file
    for filename in files:
        # Construct the new filename
        new_filename = f"{name}_000{count:d}.jpg"  # Adds leading zeros for sorting
        
        # Construct full paths for old and new filenames
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_filepath, new_filepath)
        
        # Increment the counter
        count += 1


def fetch_folder_names(parent_folder):
    # List all entries (files and folders) in the parent folder
    entries = os.listdir(parent_folder)
    
    # Initialize an empty list to store folder names
    # Iterate over each entry
    for entry in entries:
        folder_path = "G:/databasesmall/dataset/"+entry
        rename_files(folder_path,entry)  
    return 0


# Example 
parent_folder = "G:/databasesmall/dataset/"
fetch_folder_names(parent_folder)

     