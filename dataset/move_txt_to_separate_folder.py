import os
import shutil
import re

# Define the root directory (modify this to your root folder path)
ROOT_DIR = '/Users/ivanursul/Google Drive/My Drive/PhD/Dataset/Fall'

# Define the destination directory where all .tsv files will be copied
DEST_DIR = '/Users/ivanursul/Google Drive/My Drive/PhD/Dataset/Fall_sensor'

# Create the destination directory if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)

# Compile regular expressions for matching directory names
sbj_pattern = re.compile(r'^SBJ_\d+_LOC\d+$')
act_pattern = re.compile(r'^ACT\d+_[BFLR]_\d+$')

# Traverse the SBJ_* directories
for sbj_dir in os.listdir(ROOT_DIR):
    sbj_dir_path = os.path.join(ROOT_DIR, sbj_dir)
    if os.path.isdir(sbj_dir_path) and sbj_pattern.match(sbj_dir):
        # Traverse the ACT*_*_* directories within each SBJ_* directory
        for act_dir in os.listdir(sbj_dir_path):
            act_dir_path = os.path.join(sbj_dir_path, act_dir)
            if os.path.isdir(act_dir_path) and act_pattern.match(act_dir):
                # Copy all .tsv files from the ACT directories to the destination directory
                for filename in os.listdir(act_dir_path):
                    if filename.endswith('.txt'):
                        source_file = os.path.join(act_dir_path, filename)
                        shutil.copy(source_file, DEST_DIR)
                        print(f"Copied: {source_file} to {DEST_DIR}")

print(f"All .tsv files have been copied to {DEST_DIR}.")
