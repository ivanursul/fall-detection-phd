import os


def remove_stored_suffix(base_folder):
    """
    Removes '_STORED' from third-level subfolders (ACT${number2}_${side}_${number3}_STORED).

    Args:
        base_folder (str): The path to the base folder containing subject folders.
    """
    # Iterate through subject folders (SBJ_${number1})
    for subject_folder in os.listdir(base_folder):
        subject_path = os.path.join(base_folder, subject_folder)

        if os.path.isdir(subject_path) and subject_folder.startswith('SBJ'):
            # Iterate through action folders (ACT${number2}_${side}_${number3})
            for action_folder in os.listdir(subject_path):
                action_folder_path = os.path.join(subject_path, action_folder)

                if os.path.isdir(action_folder_path) and '_STORED' in action_folder:
                    # New folder name without '_STORED'
                    new_action_folder = action_folder.replace('_STORED', '')
                    new_action_folder_path = os.path.join(subject_path, new_action_folder)

                    # Rename the folder
                    print(f"Renaming {action_folder} to {new_action_folder}")
                    os.rename(action_folder_path, new_action_folder_path)


# Example usage:
base_folder = '/Users/ivanursul/Google Drive/My Drive/PhD/Dataset/Fall'  # Replace with the path to your main folder (SBJ folders)
remove_stored_suffix(base_folder)
