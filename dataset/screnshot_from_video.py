import os
import cv2
import logging

# Configure logging
logging.basicConfig(
    filename='video_screenshot.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def capture_frame_from_video(video_path, output_image_path):
    try:
        # Open the video file
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            logging.error(f"Failed to open video file: {video_path}")
            return

        # Get the total number of frames
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            logging.warning(f"Video file {video_path} has no frames.")
            return

        # Calculate the middle frame
        middle_frame = total_frames // 2
        logging.info(f"Capturing frame {middle_frame} from video: {video_path}")

        # Set the video position to the middle frame
        video.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

        # Read the middle frame
        success, frame = video.read()

        if success:
            # Save the frame as an image
            cv2.imwrite(output_image_path, frame)
            logging.info(f"Screenshot saved to {output_image_path}")
        else:
            logging.error(f"Failed to capture frame from {video_path}")

    except Exception as e:
        logging.exception(f"Error while processing video {video_path}: {e}")

    finally:
        # Release the video file
        video.release()


def process_videos_in_folder(base_folder):
    print(f"Starting to process videos in folder: {base_folder}")

    # Traverse through all folders with the prefix 'SBJ_'
    for root, dirs, files in os.walk(base_folder):
        folder_name = os.path.basename(root)

        # Check if we're in the second-level folder (folders with prefix 'SBJ_')
        if folder_name.startswith('SBJ_'):
            print(f"Processing second-level folder: {root}")

            # Iterate through the third-level subfolders
            for subdir in dirs:
                subfolder_path = os.path.join(root, subdir)
                print(f"Entering subfolder: {subfolder_path}")

                # Iterate through files in the third-level subfolder
                for sub_root, sub_subdirs, sub_files in os.walk(subfolder_path):
                    for file in sub_files:
                        if file.endswith('.mp4'):
                            video_path = os.path.join(sub_root, file)
                            logging.info(f"Found video file: {video_path}")

                            # Create the output image path (in the second-level folder)
                            output_image_path = os.path.join(root, f"{subdir}_screenshot.png")
                            logging.info(f"Saving screenshot to: {output_image_path}")

                            # Capture the middle frame and save it as a screenshot
                            capture_frame_from_video(video_path, output_image_path)

                    # Stop walking further into sub-subdirectories
                    break

    logging.info(f"Finished processing videos in folder: {base_folder}")


# Specify the base folder where the SBJ_ subfolders are located
base_folder = '/Users/ivanursul/Documents/Fall_New'

# Run the processing function
process_videos_in_folder(base_folder)

