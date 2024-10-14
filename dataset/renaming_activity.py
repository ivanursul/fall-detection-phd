

import os
import cv2

# Dictionary of options with short codes and descriptions
actions = {
    # '1': ('ACT1', 'Fall on the left'),
    # '2': ('ACT2', 'Fall on the right'),
    # '3': ('ACT3', 'Fall on the front'),
    # '4': ('ACT4', 'Fall on the back'),
    # '5': ('ACT5', 'Slide'),
    # '6': ('ACT6', 'Fall on knees'),
    # '7': ('ACT7', 'Stumble upon'),
    # '8': ('ACT8', 'Stay without walking and fall'),
    # '9': ('ACT9', 'Walk, stop and fall'),
    # '10': ('ACT10', 'Sit on chair, fall'),
    # '11': ('ACT11', 'Try to sit on chair, fall'),
    # '12': ('ACT12', 'Fall from the higher place')  # Reuses ACT1
    '13': ('ACT13', 'Walking'),
    '14': ('ACT14', 'Running'),
    '15': ('ACT15', 'Jogging'),
    '16': ('ACT16', 'Sitting'),
    '17': ('ACT17', 'Standing'),
    '18': ('ACT18', 'Picking up'),
    '19': ('ACT19', 'Laying'),
    '20': ('ACT20', 'Standing up from laying'),
    '21': ('ACT21', 'Walking, stopping, then going into another direction'),
    '22': ('ACT22', 'Waving'),
    '23': ('ACT23', 'Reaching'),
    '24': ('ACT24', 'Climbing'),
    '25': ('ACT25', 'Descending')
}

# Dictionary for directions
directions = {
    '1': ('L', 'Left'),
    '2': ('R', 'Right'),
    '3': ('F', 'Front'),
    '4': ('B', 'Back')
}

# Path to the folder containing the subfolders
root_folder = '/Users/ivanursul/Documents/Dataset/IvanMyrikZnakaDruzi/Final/SBJ_14'


# Function to generate unique folder name if it already exists
def get_unique_folder_name(base_name, root_folder):
    new_folder_path = os.path.join(root_folder, base_name)
    index = 1
    # Check if the folder exists and add index if necessary
    while os.path.exists(new_folder_path):
        new_folder_path = os.path.join(root_folder, f"{base_name}_{index}")
        index += 1
    return new_folder_path


# Loop through the subfolders
for subfolder in sorted(os.listdir(root_folder)):
    subfolder_path = os.path.join(root_folder, subfolder)

    # Check if the item is a directory and does not already start with 'ACT'
    if os.path.isdir(subfolder_path) and not subfolder.startswith('ACT'):
        # Loop through files in the subfolder to find MP4 files
        for file in os.listdir(subfolder_path):
            if file.endswith('.mp4'):
                video_path = os.path.join(subfolder_path, file)

                # Open video using OpenCV
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error opening video file {file}")
                    continue

                print(f"Showing video: {video_path}")

                speed_factor = 5

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imshow('Video', frame)

                    if cv2.waitKey(int(30 / speed_factor)) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()

                # Add a small delay to ensure windows close
                cv2.waitKey(1)

                # Display action options to the user with descriptions
                print("Choose an action type for this video:")
                for key, (code, description) in actions.items():
                    print(f"{key}. {description} ({code})")

                action_choice = input("Enter the number corresponding to the action type: ").strip()

                # Display direction options to the user
                print("Choose the direction for this action:")
                for key, (code, description) in directions.items():
                    print(f"{key}. {description} ({code})")

                direction_choice = input("Enter the number corresponding to the direction: ").strip()

                # Validate action choice
                if action_choice in actions and direction_choice in directions:
                    action_code, _ = actions[action_choice]
                    direction_code, _ = directions[direction_choice]

                    # Create base folder name with action code and direction
                    base_folder_name = f"{action_code}_{direction_code}"

                    # Get a unique folder name by adding an index if necessary
                    new_folder_path = get_unique_folder_name(base_folder_name, root_folder)

                    # Rename the folder
                    os.rename(subfolder_path, new_folder_path)
                    print(f"Renamed folder '{subfolder}' to '{new_folder_path}'")
                else:
                    print("Invalid choice for action or direction. Skipping this folder.")

                # Break after processing the first mp4 file in the folder
                break
