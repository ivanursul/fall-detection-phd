import os


def calculate_txt_files_size(folder_path):
    total_size = 0
    # Traverse through each subject folder
    for subject_folder in os.listdir(folder_path):
        subject_path = os.path.join(folder_path, subject_folder)
        if os.path.isdir(subject_path):
            # Traverse through each action subfolder
            for action_folder in os.listdir(subject_path):
                action_path = os.path.join(subject_path, action_folder)
                if os.path.isdir(action_path):
                    # Check for .txt file in the action folder
                    for file in os.listdir(action_path):
                        if file.endswith('.txt'):
                            file_path = os.path.join(action_path, file)
                            total_size += os.path.getsize(file_path)

    return total_size / (1024 * 1024)  # Convert bytes to megabytes


# Replace 'your_folder_path' with the actual path to your folder
folder_path = '/Users/ivanursul/Google Drive/My Drive/PhD/Dataset/ADL'
total_txt_size_mb = calculate_txt_files_size(folder_path)
print(f"Total size of .txt files: {total_txt_size_mb:.2f} MB")
