import os
import sys

def count_subfolders_with_mp4_and_txt(root_folder):
    count = 0
    # Traverse the first level of subdirectories
    for level1_name in os.listdir(root_folder):
        level1_path = os.path.join(root_folder, level1_name)
        if os.path.isdir(level1_path):
            # Traverse the second level of subdirectories
            for level2_name in os.listdir(level1_path):
                level2_path = os.path.join(level1_path, level2_name)
                if os.path.isdir(level2_path):
                    has_mp4 = False
                    has_txt = False
                    # Check files in the third-level subdirectory
                    for item in os.listdir(level2_path):
                        item_path = os.path.join(level2_path, item)
                        if os.path.isfile(item_path):
                            if item.lower().endswith('.mp4'):
                                has_mp4 = True
                            elif item.lower().endswith('.txt'):
                                has_txt = True
                    if has_mp4 and has_txt:
                        count += 1
    print(f"Number of third-level subfolders with both mp4 and txt files: {count}")

if __name__ == '__main__':
    root_folder = '/Users/ivanursul/Google Drive/My Drive/PhD/Dataset/ADL'
    if not os.path.isdir(root_folder):
        print(f"The path {root_folder} is not a valid directory.")
        sys.exit(1)
    count_subfolders_with_mp4_and_txt(root_folder)