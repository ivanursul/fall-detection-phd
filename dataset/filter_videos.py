import os
import shutil
import cv2


def display_mp4(mp4_path, speed_multiplier=2):
    cap = cv2.VideoCapture(mp4_path)

    if not cap.isOpened():
        print(f"Error: Couldn't open video file {mp4_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the original FPS of the video
    frame_skip = int(speed_multiplier)  # How many frames to skip for 4x speed

    while cap.isOpened():
        for _ in range(frame_skip):  # Skip frames to speed up the video
            ret, frame = cap.read()
            if not ret:
                break

        # Display the next frame
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video', frame)

        # Adjust delay to simulate faster playback
        delay = int(1000 / (fps * speed_multiplier))

        # Press 'q' to exit video display early
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Release the video capture and close the display window
    cap.release()
    cv2.destroyAllWindows()
    # Add a small delay to ensure windows close
    cv2.waitKey(1)

    print(f"Finished playing {mp4_path}")


def calculate_total_folders(base_path):
    total_folders = 0
    processed_folders = 0

    for subject_folder in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject_folder)

        if os.path.isdir(subject_path) and subject_folder.startswith('SBJ'):
            for action_folder in os.listdir(subject_path):
                action_path = os.path.join(subject_path, action_folder)
                if os.path.isdir(action_path) and action_folder.startswith('ACT'):
                    total_folders += 1
                    # Count already processed folders (STORED or DELETED)
                    if '_STORED' in action_folder or '_DELETED' in action_folder:
                        processed_folders += 1
    return total_folders, processed_folders


def process_folders(base_path):
    total_folders, processed_folders = calculate_total_folders(base_path)

    # Iterate over SBJ folders
    for subject_folder in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject_folder)

        if os.path.isdir(subject_path) and subject_folder.startswith('SBJ'):
            # Iterate over ACT folders within SBJ folder
            for action_folder in os.listdir(subject_path):
                action_path = os.path.join(subject_path, action_folder)

                if os.path.isdir(action_path) and action_folder.startswith('ACT'):
                    # Skip if the folder is already marked as _STORED or _DELETED
                    if '_STORED' in action_folder or '_DELETED' in action_folder:
                        print(f"Skipping {action_folder} (already marked)")
                        continue

                    # Find the MP4 file
                    for file in os.listdir(action_path):
                        if file.endswith('.mp4'):
                            mp4_path = os.path.join(action_path, file)

                            # Display the MP4 file
                            print(f"Showing video: {mp4_path}")
                            display_mp4(mp4_path)


                            # Ask user if they want to keep or delete the folder
                            user_choice = input(
                                f"Do you want to store or delete the folder {action_folder}? (store/delete): ").strip().lower()

                            if user_choice == 'd':
                                # Rename the folder by adding '_DELETED' to its name
                                new_folder_name = action_folder + '_DELETED'
                                new_folder_path = os.path.join(subject_path, new_folder_name)
                                os.rename(action_path, new_folder_path)
                                print(f"Renamed {action_folder} to {new_folder_name}")
                            elif user_choice == 's':
                                # Rename the folder by adding '_STORED' to its name
                                new_folder_name = action_folder + '_STORED'
                                new_folder_path = os.path.join(subject_path, new_folder_name)
                                os.rename(action_path, new_folder_path)
                                print(f"Renamed {action_folder} to {new_folder_name}")
                            else:
                                print(f"Invalid choice, skipping folder {action_folder}")

                            processed_folders += 1
                            percentage_done = (processed_folders / total_folders) * 100
                            print(f"Progress: {processed_folders}/{total_folders} folders processed ({percentage_done:.2f}%)")

if __name__ == "__main__":
    base_folder_path = '/Users/ivanursul/Google Drive/My Drive/PhD/Dataset/Fall'
    process_folders(base_folder_path)
