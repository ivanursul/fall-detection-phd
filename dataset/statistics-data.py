import os
import re
from collections import defaultdict


def parse_action_folder(folder_name):
    """
    Parse the action folder name.
    Expected formats:
      - 'ACT<number>_<side>'
      - 'ACT<number>_<side>_<number>'
    """
    match = re.match(r'^ACT(\d+)_(B|F|L|R)(?:_(\d+))?$', folder_name)
    if match:
        action_number = int(match.group(1))
        side = match.group(2)
        number3 = match.group(3)  # May be None
        return action_number, side, number3
    else:
        return None

def main():
    root_dir = '/Users/ivanursul/Google Drive/My Drive/PhD/Dataset/ADL'  # Set to your root directory containing 'SBJ_<number>' folders

    # Collect all subject folders
    subjects = [d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))]

    subject_action_counts = {}
    total_action_counts = defaultdict(int)

    for subject in subjects:
        subject_path = os.path.join(root_dir, subject)
        action_counts = defaultdict(int)
        for folder in os.listdir(subject_path):
            folder_path = os.path.join(subject_path, folder)
            if os.path.isdir(folder_path):
                parsed = parse_action_folder(folder)
                if parsed:
                    action_number, side, number3 = parsed
                    action_counts[action_number] += 1
                    total_action_counts[action_number] += 1
        subject_action_counts[subject] = action_counts

    # Print statistics of different actions for every subject
    for subject, counts in subject_action_counts.items():
        total_actions = sum(counts.values())
        print(f"Subject {subject} action counts (Total actions: {total_actions}):")
        for action in sorted(counts):
            print(f"  Action {action}: {counts[action]}")
        print()

    # Print total number of actions per subject
    print("Total number of actions per subject:")
    for subject, counts in subject_action_counts.items():
        total_actions = sum(counts.values())
        print(f"  {subject}: {total_actions}")
    print()

    # Print statistics for action per all subjects
    print("Total action counts across all subjects:")
    for action in sorted(total_action_counts):
        print(f"  Action {action}: {total_action_counts[action]}")

if __name__ == '__main__':
    main()
