import os
import shutil
from datetime import datetime

def get_timestamp(filename):
    basename = os.path.splitext(filename)[0]  # Remove the file extension
    return datetime.strptime(basename, '%Y%m%d%H%M%S')

def main():
    # Hardcoded directory path
    directory = '/Users/ivanursul/Documents/Dataset/IvanMyrikZnakaDruzi/Final/SBJ_14'  # Replace this with your actual directory path

    # Get all files in the specified directory
    files = os.listdir(directory)

    # Filter out only .mp4 and .txt files
    relevant_files = [f for f in files if f.endswith('.mp4') or f.endswith('.txt')]

    # Create a list of files with their timestamps
    ungrouped_files = []
    for f in relevant_files:
        try:
            timestamp = get_timestamp(f)
            ungrouped_files.append({'filename': f, 'timestamp': timestamp})
        except ValueError:
            print(f"Filename {f} does not match the expected format and will be skipped.")

    # Sort files by timestamp
    ungrouped_files.sort(key=lambda x: x['timestamp'])

    groups = []
    i = 0
    n = len(ungrouped_files)

    # Group files within 5 seconds of each other
    while i < n:
        current_group = [ungrouped_files[i]]
        j = i + 1
        while j < n:
            time_diff = (ungrouped_files[j]['timestamp'] - ungrouped_files[i]['timestamp']).total_seconds()
            if time_diff <= 5:
                current_group.append(ungrouped_files[j])
                j += 1
            else:
                break
        groups.append(current_group)
        i = j

    # Create subfolders and move files
    for group in groups:
        if len(group) > 1:
            earliest_timestamp = min(f['timestamp'] for f in group)
            # Folder name format with underscores
            folder_name = earliest_timestamp.strftime('%Y_%m_%d_%H_%M_%S')
            folder_path = os.path.join(directory, folder_name)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            for f in group:
                src_path = os.path.join(directory, f['filename'])
                dst_path = os.path.join(folder_path, f['filename'])
                shutil.move(src_path, dst_path)
        else:
            # Do not create a folder if there's only one file
            pass

if __name__ == '__main__':
    main()
