import os

# Specify the path where you want to create the folders
parent_directory = "/Users/ivanursul/Documents/Dataset/ADL/Drapey/Final"

# Create folders from 1 to 50 inside the specified directory
for i in range(1, 55):
    folder_name = str(i)
    folder_path = os.path.join(parent_directory, folder_name)
    # Check if the folder already exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")