import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

def apply_moving_average_filter(data, window_size=5):
    # Apply moving average filter to each accelerometer axis
    data['AccX(g)'] = data['AccX(g)'].rolling(window=window_size, center=True).mean()
    data['AccY(g)'] = data['AccY(g)'].rolling(window=window_size, center=True).mean()
    data['AccZ(g)'] = data['AccZ(g)'].rolling(window=window_size, center=True).mean()
    return data


def apply_butterworth_filter(data, cutoff=5, fs=50, order=2):
    # Design Butterworth filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply filter to each axis
    data['AccX(g)'] = filtfilt(b, a, data['AccX(g)'])
    data['AccY(g)'] = filtfilt(b, a, data['AccY(g)'])
    data['AccZ(g)'] = filtfilt(b, a, data['AccZ(g)'])
    return data


def apply_savgol_filter(data, window_length=11, polyorder=2):
    # Apply Savitzky-Golay filter to each axis
    data['AccX(g)'] = savgol_filter(data['AccX(g)'], window_length, polyorder)
    data['AccY(g)'] = savgol_filter(data['AccY(g)'], window_length, polyorder)
    data['AccZ(g)'] = savgol_filter(data['AccZ(g)'], window_length, polyorder)
    return data

# Function to calculate accelerometer angles (roll, pitch)
def calculate_acc_angles(accX, accY, accZ):
    roll_acc = np.arctan2(accY, np.sqrt(accX**2 + accZ**2))
    pitch_acc = np.arctan2(-accX, np.sqrt(accY**2 + accZ**2))
    yaw_acc = np.zeros_like(roll_acc)  # Placeholder, yaw can't be calculated from accelerometer
    return roll_acc, pitch_acc, yaw_acc

# Function to integrate gyroscope data (roll, pitch, yaw)
def integrate_gyro(gyroX, gyroY, gyroZ, dt):
    roll_gyro = np.cumsum(gyroX * dt)
    pitch_gyro = np.cumsum(gyroY * dt)
    yaw_gyro = np.cumsum(gyroZ * dt)
    return roll_gyro, pitch_gyro, yaw_gyro

# Complementary filter to combine accelerometer and gyroscope data
def apply_complementary_filter(acc_angle, gyro_angle, alpha=0.98):
    return alpha * gyro_angle + (1 - alpha) * acc_angle


# Function to correct accelerometer data based on roll, pitch, yaw
def correct_accelerometer(accX, accY, accZ, roll, pitch, yaw):
    # Rotation matrices
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R_combined = R_yaw @ R_pitch @ R_roll

    # Original accelerometer vector
    acc_vector = np.array([accX, accY, accZ])

    # Rotate accelerometer vector
    acc_corrected = R_combined @ acc_vector

    return acc_corrected



def adjust_fall_activity_labels(data):
    # Create a boolean mask where Activity is 'Fall'
    data['Is_Fall'] = data['Activity'] == 'Fall'

    # Identify where 'Is_Fall' changes to find consecutive groups
    data['Fall_Group'] = (data['Is_Fall'] != data['Is_Fall'].shift()).cumsum()

    # Keep only the 'Fall' periods
    fall_groups = data[data['Is_Fall']]

    # Compute the length of each fall group
    group_lengths = fall_groups.groupby('Fall_Group').size()

    # Find the group with the maximum length
    if not group_lengths.empty:
        longest_fall_group = group_lengths.idxmax()

        # Relabel all 'Fall' periods that are not the longest to 'Non-fall'
        fall_groups_to_relabel = fall_groups[fall_groups['Fall_Group'] != longest_fall_group]
        data.loc[fall_groups_to_relabel.index, 'Activity'] = 'Non-fall'

    # Clean up temporary columns
    data.drop(['Is_Fall', 'Fall_Group'], axis=1, inplace=True)

    return data


def preprocess_and_analyze_data(input_folder, output_folder,
                                non_fall_threshold=1.69, fall_threshold=1.7,
                                window_size=2, cutoff=2, fs=20, order=2, sampling_rate=50):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each CSV file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_folder, file_name)
            data = pd.read_csv(file_path, delimiter='\t')
            data['time_formatted'] = data['time'].apply(lambda x: x[::-1].replace(':', '.', 1)[::-1])
            data['time_formatted'] = pd.to_datetime(data['time_formatted'])


            # Correct AccZ(g) by subtracting 1g to remove gravity
            data['Corrected_AccZ'] = data['AccZ(g)'] - 1

            # Calculate magnitude of corrected acceleration
            data['Corrected_Magnitude'] = np.sqrt(
                data['AccX(g)'] ** 2 + data['AccY(g)'] ** 2 + data['Corrected_AccZ'] ** 2)

            # Initialize all data as Rest
            data['Activity'] = 'Non-fall'

            # Detect standing: very low corrected magnitude (still or minimal movement)
            data.loc[data['Corrected_Magnitude'] <= non_fall_threshold, 'Activity'] = 'Non-fall'

            # Detect fall: sudden spike in corrected magnitude above the fall threshold
            data.loc[data['Corrected_Magnitude'] >= fall_threshold, 'Activity'] = 'Fall'

            # Adjust activity labels to keep 'Fall' only for the longest period
            data = adjust_fall_activity_labels(data)

            # Plot the graph for each CSV
            plot_activity_graph(data, file_name, output_folder)


def plot_activity_graph(data, file_name, output_folder):
    # Set up the plot
    plt.figure(figsize=(14, 8))

    # Plot AccX(g), AccY(g), AccZ(g)
    plt.subplot(3, 1, 1)
    plt.plot(data['time'], data['AccX(g)'], label='AccX(g)', color='red', alpha=0.7)
    plt.plot(data['time'], data['AccY(g)'], label='AccY(g)', color='green', alpha=0.7)
    plt.plot(data['time'], data['AccZ(g)'], label='AccZ(g)', color='blue', alpha=0.7)
    plt.ylabel('Acceleration (g)')
    plt.legend(loc='upper right')
    plt.title(f'Acceleration Data for {file_name}')

    # Plot Magnitude
    plt.subplot(3, 1, 2)
    plt.plot(data['time'], data['Corrected_Magnitude'], label='Corrected Magnitude', color='black')
    plt.ylabel('Magnitude')
    plt.legend(loc='upper right')

    # Plot activities (color-coded)
    plt.subplot(3, 1, 3)
    plt.plot(data['time'], data['Corrected_Magnitude'], label='Corrected Magnitude', color='black')

    plt.fill_between(data['time'], data['Corrected_Magnitude'], where=data['Activity'] == 'Non-fall', color='green',
                     alpha=0.3, label='Non-fall')
    plt.fill_between(data['time'], data['Corrected_Magnitude'], where=data['Activity'] == 'Fall', color='red',
                     alpha=0.6, label='Fall')

    plt.xlabel('Time (s)')
    plt.ylabel('Corrected Magnitude')
    plt.legend(loc='upper right')

    # Save the plot to a file
    output_path = os.path.join(output_folder, f'{file_name}_activity_plot.png')
    plt.savefig(output_path)
    plt.close()

# Example usage:
input_folder = "/Users/ivanursul/Google Drive/My Drive/PhD/Dataset/Sensor Data/Fall"
output_folder = "/Users/ivanursul/Google Drive/My Drive/PhD/Dataset/Sensor Data/Fall_plot"
preprocess_and_analyze_data(input_folder, output_folder)
