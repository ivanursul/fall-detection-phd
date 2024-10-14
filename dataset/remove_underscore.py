import os
import shutil

# Source and destination folder paths
source_folder = '/Users/ivanursul/Documents/Dataset/IvanMyrikZnakaDruzi/Raw Videos'
destination_folder = '/Users/ivanursul/Documents/Dataset/IvanMyrikZnakaDruzi/Merged'

# Make sure the destination folder exists, if not, create it
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Loop through all files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file is a .mp4 video file
    if filename.endswith('.mp4'):
        # Remove underscores from the filename
        new_filename = filename.replace('_', '')

        # Full paths for source and destination files
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, new_filename)

        # Copy the file to the destination with the new name
        shutil.copy(source_file, destination_file)
        print(f'Copied: {filename} -> {new_filename}')

print("All videos copied and renamed successfully!")