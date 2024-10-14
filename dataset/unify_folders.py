import os
import re


def generate_new_number(existing_numbers):
    """Generates a new number not in the list of existing_numbers."""
    new_number = 1
    while new_number in existing_numbers:
        new_number += 1
    return new_number


def get_existing_action_numbers(subfolders, action_number, side):
    """Extracts the existing action numbers for a specific ACT number and side."""
    existing_numbers = []
    for folder in subfolders:
        match = re.match(rf'ACT{action_number}_{side}(_\d+)?', folder)
        if match and match.group(1):  # Check if the number suffix exists
            existing_numbers.append(int(match.group(1)[1:]))  # Extract the suffix number
    return existing_numbers


def rename_folders(base_folder):
    """Renames the action subfolders to the unified format."""
    for subject_folder in os.listdir(base_folder):
        subject_path = os.path.join(base_folder, subject_folder)
        if os.path.isdir(subject_path) and subject_folder.startswith('SBJ'):
            action_subfolders = os.listdir(subject_path)

            for action_folder in action_subfolders:
                action_path = os.path.join(subject_path, action_folder)
                if os.path.isdir(action_path):
                    # Check if the folder is in 'ACT${number2}_${side}' or 'ACT${number2}_${side}_${number3}' format
                    match = re.match(r'ACT(\d+)_([BFRL])(_\d+)?', action_folder)
                    if match:
                        action_number = int(match.group(1))
                        side = match.group(2)
                        number_suffix = match.group(3)

                        # Get existing numbers for this action number and side
                        existing_numbers = get_existing_action_numbers(action_subfolders, action_number, side)

                        if number_suffix is None:  # If there's no number suffix
                            # Generate a new number that isn't used yet for this action_number and side
                            new_number = generate_new_number(existing_numbers)
                            existing_numbers.append(new_number)

                            # Rename the folder
                            new_name = f'ACT{action_number}_{side}_{new_number}'
                            new_path = os.path.join(subject_path, new_name)
                            print(f"Renaming {action_folder} to {new_name}")
                            os.rename(action_path, new_path)
                    else:
                        print(f"Skipping {action_folder}: doesn't match format")


# Example usage:
base_folder = '/Users/ivanursul/Documents/Dataset/IvanMyrikZnakaDruzi/Final'  # Replace with the path to your main folder
rename_folders(base_folder)
