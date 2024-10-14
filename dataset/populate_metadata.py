import os
import json
import re

def create_metadata_json(base_folder):
    """
    Creates a metadata.json file in every ACT${number2}_${side}_${number3} folder.

    Args:
        base_folder (str): The path to the base folder containing subject folders.
    """
    # Iterate through subject folders (SBJ_${number1}_LOC${number11})
    for subject_folder in os.listdir(base_folder):
        subject_match = re.match(r'SBJ_(\d+)_LOC(\d+)', subject_folder)
        if subject_match:
            subject_id = int(subject_match.group(1))
            location_id = int(subject_match.group(2))
            subject_path = os.path.join(base_folder, subject_folder)

            if os.path.isdir(subject_path):
                # Iterate through action folders (ACT${number2}_${side}_${number3})
                for action_folder in os.listdir(subject_path):
                    action_match = re.match(r'ACT(\d+)_([BFRL])_(\d+)', action_folder)
                    if action_match:
                        action_id = int(action_match.group(1))
                        side = action_match.group(2)
                        attempt = int(action_match.group(3))
                        action_folder_path = os.path.join(subject_path, action_folder)

                        if os.path.isdir(action_folder_path):
                            # Create the metadata
                            metadata = {
                                "subjectId": subject_id,
                                "locationId": location_id,
                                "actionId": action_id,
                                "side": side,
                                "attempt": attempt
                            }

                            # Path to the metadata.json file
                            metadata_file_path = os.path.join(action_folder_path, 'metadata.json')

                            # Write the metadata to the file in JSON format
                            with open(metadata_file_path, 'w') as metadata_file:
                                json.dump(metadata, metadata_file, indent=4)

                            print(f"Created metadata.json in {action_folder_path}")

# Example usage:
base_folder = '/Users/ivanursul/Google Drive/My Drive/PhD/Dataset/Fall'  # Replace with the path to your main folder (SBJ folders)
create_metadata_json(base_folder)
