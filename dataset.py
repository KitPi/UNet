# Remove the subfolders in training dataset

import os
import shutil

def merge_subfolders(main_folder):
    # Iterate through all items in the main folder
    for item in os.listdir(main_folder):
        item_path = os.path.join(main_folder, item)
        # Check if the item is a directory (subfolder)
        if os.path.isdir(item_path):
            # Move all files from the subfolder to the main folder
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                # Move the file to the main folder
                shutil.move(sub_item_path, main_folder)
                print(f"Moved: {sub_item_path} to {main_folder}")
            # Remove the now-empty subfolder
            os.rmdir(item_path)
            print(f"Removed subfolder: {item_path}")

# Specify the path to your main folder
main_folder_path = 'dataset/train/'

# Call the function to merge subfolder contents
merge_subfolders(main_folder_path)