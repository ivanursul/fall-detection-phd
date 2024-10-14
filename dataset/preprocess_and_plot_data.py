import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt,find_peaks


# Kalman Filter class
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_estimate, initial_error):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_estimate
        self.error = initial_error
        self.kalman_gain = 0

    def update(self, measurement):
        # Update the Kalman Gain
        self.kalman_gain = self.error / (self.error + self.measurement_variance)
        # Update the estimate with the measurement
        self.estimate = self.estimate + self.kalman_gain * (measurement - self.estimate)
        # Update the error estimate
        self.error = (1 - self.kalman_gain) * self.error + np.abs(self.estimate) * self.process_variance
        return self.estimate


def filter_data_from_noise_complementary_filter(df, alpha=0.8):
    """
    Apply Complementary Filter to reduce noise in accelerometer data using gyroscope data.

    :param df: Input DataFrame containing accelerometer and gyroscope data
    :param alpha: Weighting factor for the complementary filter (default 0.98)
    :return: DataFrame with filtered accelerometer data
    """
    # Initialize filtered columns with the original accelerometer data
    df['AccX_filtered'] = df['AccX(g)']
    df['AccY_filtered'] = df['AccY(g)']
    df['AccZ_filtered'] = df['AccZ(g)']

    # Get the sampling rate based on the time column (assuming time is in seconds)
    df['time_formatted'] = df['time'].apply(lambda x: x[::-1].replace(':', '.', 1)[::-1])
    df['time_formatted'] = pd.to_datetime(df['time_formatted'])
    df['time_seconds'] = (df['time_formatted'] - df['time_formatted'].iloc[0]).dt.total_seconds()

    time = df['time_seconds'].values
    dt = np.mean(np.diff(time))  # Time step between samples

    # Initialize filtered values (starting from the first row values)
    acc_x_filtered = df['AccX(g)'].iloc[0]
    acc_y_filtered = df['AccY(g)'].iloc[0]
    acc_z_filtered = df['AccZ(g)'].iloc[0]

    # Apply the complementary filter to each row
    for i in range(1, len(df)):
        # Gyroscope contributes to short-term estimation of velocity (rate of change in acceleration)
        gyro_x = df['AsX(°/s)'].iloc[i]
        gyro_y = df['AsY(°/s)'].iloc[i]
        gyro_z = df['AsZ(°/s)'].iloc[i]

        # Update accelerometer estimates using complementary filter
        acc_x_filtered = alpha * (acc_x_filtered + gyro_x * dt) + (1 - alpha) * df['AccX(g)'].iloc[i]
        acc_y_filtered = alpha * (acc_y_filtered + gyro_y * dt) + (1 - alpha) * df['AccY(g)'].iloc[i]
        acc_z_filtered = alpha * (acc_z_filtered + gyro_z * dt) + (1 - alpha) * df['AccZ(g)'].iloc[i]

        # Save filtered values
        df.at[i, 'AccX_filtered'] = acc_x_filtered
        df.at[i, 'AccY_filtered'] = acc_y_filtered
        df.at[i, 'AccZ_filtered'] = acc_z_filtered

    return df


def filter_data_from_noise_kalman_filter(df):
    # Initialize Kalman filters for accelerometer and gyroscope data
    kf_acc_x = KalmanFilter(process_variance=1e-5, measurement_variance=1e-2, initial_estimate=df['AccX(g)'].iloc[0], initial_error=1)
    kf_acc_y = KalmanFilter(process_variance=1e-5, measurement_variance=1e-2, initial_estimate=df['AccY(g)'].iloc[0], initial_error=1)
    kf_acc_z = KalmanFilter(process_variance=1e-5, measurement_variance=1e-2, initial_estimate=df['AccZ(g)'].iloc[0], initial_error=1)

    # Apply Kalman filter to accelerometer and gyroscope data
    df['AccX_filtered2'] = df['AccX(g)'].apply(kf_acc_x.update)
    df['AccY_filtered2'] = df['AccY(g)'].apply(kf_acc_y.update)
    df['AccZ_filtered2'] = df['AccZ(g)'].apply(kf_acc_z.update)

    return df


# Define a Butterworth filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    padlen = order * 3  # Minimum length required for filtfilt

    if len(data) < padlen:
        # If the data is too short, skip filtering or use an alternative filter
        print(f"Segment too short for filtfilt (length: {len(data)}), using original data")
        return data  # Optionally, apply a simple moving average filter here instead

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    try:
        # Attempt to apply filtfilt, ensuring padlen condition is met
        y = filtfilt(b, a, data)
    except ValueError as e:
        print(f"Error during filtering: {e}")
        return data  # Return the original data in case of any errors

    return y

def calculate_derived_values(df):
    df['AccX_original'] = df['AccX(g)']
    df['AccY_original'] = df['AccY(g)']
    df['AccZ_original'] = df['AccZ(g)']

    """
    Calculates derived values such as magnitude of acceleration and angular speed.

    :param df: Input DataFrame containing accelerometer and gyroscope data.
    :return: DataFrame with derived values added.
    """
    # Calculate the magnitude of the accelerometer (AccX, AccY, AccZ)
    df['Acc_magnitude(g)'] = np.sqrt(df['AccX(g)']**2 + df['AccY(g)']**2 + df['AccZ(g)']**2)

    # Calculate the magnitude of the gyroscope (AsX, AsY, AsZ)
    df['As_magnitude(°/s)'] = np.sqrt(df['AsX(°/s)']**2 + df['AsY(°/s)']**2 + df['AsZ(°/s)']**2)

    # Example of adding tilt angle (calculated from acceleration):
    df['tilt_angle(deg)'] = np.arctan2(df['AccY(g)'], np.sqrt(df['AccX(g)']**2 + df['AccZ(g)']**2)) * (180 / np.pi)

    return df

def calculate_additional_data_for_filtered_results(df):
    """
    Calculates derived values such as magnitude of acceleration and angular speed.

    :param df: Input DataFrame containing accelerometer and gyroscope data.
    :return: DataFrame with derived values added.
    """

    df['Corrected_AccZ_i'] = df['AccZ_filtered_i'] - 1

    # Calculate the magnitude of the accelerometer (AccX, AccY, AccZ)
    df['Acc_magnitude_i'] = np.sqrt(df['AccX_filtered_i']**2 + df['AccY_filtered_i']**2 + df['Corrected_AccZ_i']**2)

    # Calculate the magnitude of the gyroscope (AsX, AsY, AsZ)
    df['As_magnitude_i'] = np.sqrt(df['AsX(°/s)']**2 + df['AsY(°/s)']**2 + df['AsZ(°/s)']**2)

    # Example of adding tilt angle (calculated from acceleration):
    df['tilt_angle_i'] = np.arctan2(df['AccY(g)'], np.sqrt(df['AccX(g)']**2 + df['AccZ(g)']**2)) * (180 / np.pi)

    return df


# Function to apply noise filtering on non-fall segments
def filter_non_fall_segments(df, fall_start, fall_end, cutoff=5, fs=100, order=4):
    # Initialize filtered columns with original data
    df['AccX_filtered_i'] = df['AccX(g)']
    df['AccY_filtered_i'] = df['AccY(g)']
    df['AccZ_filtered_i'] = df['AccZ(g)']

    # Apply filtering only to the segments outside of the fall event
    if fall_start is not None and fall_end is not None:
        # Filter the data before the fall start
        if fall_start > 0:
            filtered_accX_pre_fall = butter_lowpass_filter(df['AccX(g)'].iloc[:fall_start], cutoff, fs, order)
            filtered_accY_pre_fall = butter_lowpass_filter(df['AccY(g)'].iloc[:fall_start], cutoff, fs, order)
            filtered_accZ_pre_fall = butter_lowpass_filter(df['AccZ(g)'].iloc[:fall_start], cutoff, fs, order)

            # Assign the filtered data back to the corresponding segments
            df.loc[:fall_start - 1, 'AccX_filtered_i'] = filtered_accX_pre_fall
            df.loc[:fall_start - 1, 'AccY_filtered_i'] = filtered_accY_pre_fall
            df.loc[:fall_start - 1, 'AccZ_filtered_i'] = filtered_accZ_pre_fall

        # Filter the data after the fall end
        if fall_end < len(df):
            filtered_accX_post_fall = butter_lowpass_filter(df['AccX(g)'].iloc[fall_end + 1:], cutoff, fs, order)
            filtered_accY_post_fall = butter_lowpass_filter(df['AccY(g)'].iloc[fall_end + 1:], cutoff, fs, order)
            filtered_accZ_post_fall = butter_lowpass_filter(df['AccZ(g)'].iloc[fall_end + 1:], cutoff, fs, order)

            # Assign the filtered data back to the corresponding segments
            df.loc[fall_end + 1:, 'AccX_filtered_i'] = filtered_accX_post_fall
            df.loc[fall_end + 1:, 'AccY_filtered_i'] = filtered_accY_post_fall
            df.loc[fall_end + 1:, 'AccZ_filtered_i'] = filtered_accZ_post_fall

    return df


def apply_butter_lowpass_filter_to_all(df, cutoff=5, fs=100, order=4):
    """
    Apply the butter_lowpass_filter to the entire dataset for each accelerometer axis (AccX, AccY, AccZ).

    :param df: Input DataFrame with accelerometer data
    :param cutoff: Cutoff frequency for the Butterworth filter
    :param fs: Sampling frequency
    :param order: Order of the Butterworth filter
    :return: DataFrame with filtered columns for AccX, AccY, and AccZ
    """
    # Apply the Butterworth filter to each axis
    df['AccX_filtered_i'] = butter_lowpass_filter(df['AccX(g)'], cutoff, fs, order)
    df['AccY_filtered_i'] = butter_lowpass_filter(df['AccY(g)'], cutoff, fs, order)
    df['AccZ_filtered_i'] = butter_lowpass_filter(df['AccZ(g)'], cutoff, fs, order)

    return df


# Function to detect the most dense segment
def detect_fall_segment(df, window_size=10, threshold=0.5):
    # Calculate a rolling window to detect dense segments of high magnitude
    df['rolling_sum'] = df['Acc_magnitude(g)'].rolling(window=window_size).sum()

    # Find the index of the maximum rolling sum (most dense segment)
    fall_start = df['rolling_sum'].idxmax() if not df['rolling_sum'].isna().all() else None
    fall_end = fall_start + window_size if fall_start else None

    return fall_start, fall_end


def expand_fall_spike(df, fall_start, fall_end, threshold_factor=0.3):
    peak_value = df['Acc_magnitude(g)'].iloc[fall_start:fall_end].max()
    min_threshold = peak_value * threshold_factor

    # Extend backwards
    while fall_start > 0 and df['Acc_magnitude(g)'].iloc[fall_start] > min_threshold:
        fall_start -= 1

    # Extend forwards
    while fall_end < len(df) - 1 and df['Acc_magnitude(g)'].iloc[fall_end] > min_threshold:
        fall_end += 1

    return fall_start, fall_end



def detect_multiple_spikes(df, window_size=10, density_threshold=0.5):
    """
    Detects multiple spikes based on the rolling sum of accelerometer magnitudes.
    Spikes are marked if their rolling sum exceeds the given threshold.

    :param df: Input DataFrame with calculated Acc_magnitude(g)
    :param window_size: The size of the rolling window for detecting density
    :param density_threshold: The threshold for considering a segment as a spike
    :return: A list of tuples where each tuple is (start_index, end_index) of detected spikes
    """
    # Calculate a rolling window to detect dense segments of high magnitude
    df['rolling_sum'] = df['Acc_magnitude(g)'].rolling(window=window_size).sum()

    spikes = []
    in_spike = False
    spike_start = None

    for i in range(len(df)):
        if df['rolling_sum'].iloc[i] > density_threshold and not in_spike:
            # Start of a spike
            spike_start = i
            in_spike = True
        elif df['rolling_sum'].iloc[i] <= density_threshold and in_spike:
            # End of the spike
            spike_end = i
            spikes.append((spike_start, spike_end))
            in_spike = False

    # If the spike ends at the last data point
    if in_spike:
        spikes.append((spike_start, len(df) - 1))

    return spikes


def apply_butter_lowpass_filter_for_non_fall_segments(df, spikes, cutoff=5, fs=100, order=4):
    """
    Apply noise filtering on all segments except for those marked as spikes.

    :param df: Input DataFrame with accelerometer data
    :param spikes: List of tuples (start, end) representing spike regions
    :param cutoff: Cutoff frequency for the Butterworth filter
    :param fs: Sampling frequency
    :param order: Order of the Butterworth filter
    :return: DataFrame with filtered non-spike segments
    """
    # Initialize filtered columns with original data
    df['AccX_filtered_i'] = df['AccX(g)']
    df['AccY_filtered_i'] = df['AccY(g)']
    df['AccZ_filtered_i'] = df['AccZ(g)']

    # Get all non-spike regions
    last_end = 0
    for spike_start, spike_end in spikes:
        # Filter data before the spike
        if last_end < spike_start:
            df.loc[last_end:spike_start - 1, 'AccX_filtered_i'] = butter_lowpass_filter(
                df['AccX(g)'].iloc[last_end:spike_start], cutoff, fs, order)
            df.loc[last_end:spike_start - 1, 'AccY_filtered_i'] = butter_lowpass_filter(
                df['AccY(g)'].iloc[last_end:spike_start], cutoff, fs, order)
            df.loc[last_end:spike_start - 1, 'AccZ_filtered_i'] = butter_lowpass_filter(
                df['AccZ(g)'].iloc[last_end:spike_start], cutoff, fs, order)

        # After each spike, set the last end to the spike end
        last_end = spike_end + 1

    # Filter the remaining data after the last spike
    if last_end < len(df):
        df.loc[last_end:, 'AccX_filtered_i'] = butter_lowpass_filter(df['AccX(g)'].iloc[last_end:], cutoff, fs, order)
        df.loc[last_end:, 'AccY_filtered_i'] = butter_lowpass_filter(df['AccY(g)'].iloc[last_end:], cutoff, fs, order)
        df.loc[last_end:, 'AccZ_filtered_i'] = butter_lowpass_filter(df['AccZ(g)'].iloc[last_end:], cutoff, fs, order)

    return df


def expand_fall_spike_for_multiple_spikes(df, spikes, threshold_factor=0.3):
    """
    Expands each detected spike to ensure the entire fall event is captured,
    extending backward and forward based on a threshold factor.

    :param df: Input DataFrame with accelerometer magnitude values
    :param spikes: List of tuples (start, end) representing spike regions
    :param threshold_factor: Factor to determine how much to extend the spike
                             based on the peak value within the spike
    :return: List of expanded spikes with (start, end) indices
    """
    expanded_spikes = []

    for (fall_start, fall_end) in spikes:
        # Get the peak value within the current spike
        peak_value = df['Acc_magnitude(g)'].iloc[fall_start:fall_end].max()
        min_threshold = peak_value * threshold_factor

        # Extend backwards
        while fall_start > 0 and df['Acc_magnitude(g)'].iloc[fall_start] > min_threshold:
            fall_start -= 1

        # Extend forwards
        while fall_end < len(df) - 1 and df['Acc_magnitude(g)'].iloc[fall_end] > min_threshold:
            fall_end += 1

        # Add the expanded spike to the list
        expanded_spikes.append((fall_start, fall_end))

    return expanded_spikes


def apply_linear_interpolation_for_spikes(df, spike_regions, acc_x_column, acc_y_column, acc_z_column):
    # Now, remove the spike regions and interpolate the values for AccX, AccY, and AccZ
    for spike_start, spike_end, width, height in spike_regions:
        # Set the spike region to NaN for all AccX(g), AccY(g), and AccZ(g)
        df.loc[spike_start:spike_end, [acc_x_column, acc_y_column, acc_z_column]] = None

        # Perform linear interpolation for AccX(g), AccY(g), and AccZ(g)
        df[acc_x_column].interpolate(method='linear', inplace=True)
        df[acc_y_column].interpolate(method='linear', inplace=True)
        df[acc_z_column].interpolate(method='linear', inplace=True)

    # If needed, you can also fill any remaining NaN values using forward or backward filling
    df[acc_x_column].fillna(method='ffill', inplace=True)
    df[acc_y_column].fillna(method='ffill', inplace=True)
    df[acc_z_column].fillna(method='bfill', inplace=True)


def plot_and_save_data(file_path, output_folder, output_folder_csv, dataset_type, height_threshold=2.0):
    # Read the TSV file (tab-separated values)
    df = pd.read_csv(file_path, delimiter='\t')

    calculate_derived_values(df)

    filter_data_from_noise_kalman_filter(df)
    filter_data_from_noise_complementary_filter(df)

    cutoff = 5
    fs = 100
    order = 4
    window_size = 10
    fall_width_threshold = 27

    # Calculate spikes for non-filtered data
    spike_regions_raw_data = calculate_all_spikes(df, height_threshold, 'Acc_magnitude(g)')

    # Find all spikes that exceed certain width and height
    filtered_spike_regions = [
        (spike_start, spike_end, width, height)
        for spike_start, spike_end, width, height in spike_regions_raw_data
        if width <= 2 and height >= 2.0
    ]

    # Smooth spikes out
    apply_linear_interpolation_for_spikes(df, filtered_spike_regions, 'AccX(g)', 'AccY(g)', 'AccZ(g)')

    if dataset_type == "Falls":
        fall_spikes = detect_multiple_spikes(df, window_size, fall_width_threshold)
        fall_spikes = expand_fall_spike_for_multiple_spikes(df, fall_spikes, threshold_factor=0.3)
        df = apply_butter_lowpass_filter_for_non_fall_segments(df, fall_spikes, cutoff, fs)
    else:
        fall_spikes = []
        apply_butter_lowpass_filter_to_all(df, cutoff, fs, order)

    # Calculate additional data
    df = calculate_additional_data_for_filtered_results(df)

    # build and save graph
    plot_and_save_graphs(df, file_path, output_folder, spike_regions_raw_data, fall_spikes)

    # Drop non filtered data
    columns_to_keep = ['AccX_filtered_i', 'AccY_filtered_i', 'AccZ_filtered_i', 'Corrected_AccZ_i', 'Acc_magnitude_i']
    df = df[columns_to_keep]

    output_csv_path = os.path.join(output_folder_csv, f"{os.path.splitext(os.path.basename(file_path))[0]}.csv")
    df.to_csv(output_csv_path, index=False)


def plot_and_save_graphs(df, file_path, output_folder, spike_regions, fall_spikes):
    # Create a figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2 rows, 2 columns

    # Assign indices for ease of access
    raw_data_idx = (0, 0)
    raw_magnitude_idx = (0, 1)
    processed_data_idx = (1, 0)
    processed_magnitude_idx = (1, 1)

    # Plot accelerometer data for X, Y, and Z axes (raw data) on the first subplot
    axs[raw_data_idx].plot(df.index, df['AccX_original'], label='AccX(g)', color='r')
    axs[raw_data_idx].plot(df.index, df['AccY_original'], label='AccY(g)', color='g')
    axs[raw_data_idx].plot(df.index, df['AccZ_original'], label='AccZ(g)', color='b')
    axs[raw_data_idx].set_title(f'Raw Data - {os.path.basename(file_path)}')
    axs[raw_data_idx].set_xlabel('Time')
    axs[raw_data_idx].set_ylabel('Acceleration (g)')
    axs[raw_data_idx].legend()
    axs[raw_data_idx].grid(True)

    # Highlight spike regions and annotate the width (number of records)
    for start, end, width, height in spike_regions:
        axs[raw_data_idx].axvspan(start, end, color='yellow', alpha=0.5)  # Highlight region
        mid_point = (start + end) // 2
        axs[raw_data_idx].text(mid_point, df[['AccX_original', 'AccY_original', 'AccZ_original']].iloc[start:end].max().max() + 0.05,
                               f'{width} records', color='black', ha='center')

    # Plot raw acceleration magnitude
    axs[raw_magnitude_idx].plot(df.index, df['Acc_magnitude(g)'], label='Acc Magnitude', color='purple')
    axs[raw_magnitude_idx].set_title('Raw Acceleration Magnitude')
    axs[raw_magnitude_idx].set_xlabel('Record Number')
    axs[raw_magnitude_idx].set_ylabel('Magnitude (g)')
    axs[raw_magnitude_idx].legend()
    axs[raw_magnitude_idx].grid(True)

    # Plot accelerometer data filtered by Ivan (processed data) on the third subplot
    axs[processed_data_idx].plot(df.index, df['AccX_filtered_i'], label='AccX_filtered_i', color='r')
    axs[processed_data_idx].plot(df.index, df['AccY_filtered_i'], label='AccY_filtered_i', color='g')
    axs[processed_data_idx].plot(df.index, df['AccZ_filtered_i'], label='AccZ_filtered_i', color='b')
    axs[processed_data_idx].set_title('Processed Data')
    axs[processed_data_idx].set_xlabel('Time')
    axs[processed_data_idx].set_ylabel('Acceleration (g)')
    axs[processed_data_idx].legend()
    axs[processed_data_idx].grid(True)

    # Highlight the spikes on the processed data graph
    for spike_start, spike_end in fall_spikes:
        axs[processed_data_idx].axvspan(spike_start, spike_end, color='green', alpha=0.5)  # Highlight region

    # Plot processed acceleration magnitude
    axs[processed_magnitude_idx].plot(df.index, df['Acc_magnitude_i'], label='Processed Acc Magnitude', color='purple')
    axs[processed_magnitude_idx].set_title('Processed Acceleration Magnitude')
    axs[processed_magnitude_idx].set_xlabel('Record Number')
    axs[processed_magnitude_idx].set_ylabel('Magnitude (g)')
    axs[processed_magnitude_idx].legend()
    axs[processed_magnitude_idx].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the combined plot as an image in the output folder
    output_image_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}.png")
    plt.savefig(output_image_path)
    plt.close()

    print(f"Saved plot to: {output_image_path}")


def calculate_all_spikes(df, spike_threshold, magnitude_field):
    # Initialize variables
    spike_regions = []
    in_spike = False
    spike_start = None

    # Identify where the magnitude crosses the threshold and track spikes
    for i in range(1, len(df)):
        if df[magnitude_field].iloc[i] > spike_threshold and not in_spike:
            # Spike starts
            spike_start = i
            in_spike = True
        elif df[magnitude_field].iloc[i] <= spike_threshold and in_spike:
            # Spike ends
            spike_end = i
            width = spike_end - spike_start

            # Calculate the height of the spike (difference between max and min values during the spike)
            spike_data = df[magnitude_field].iloc[spike_start:spike_end]
            height = spike_data.max()

            # Filter based on width <= 2 and height >= 2
            # if width <= 2 and height >= 2.0:
            #     spike_regions.append((spike_start, spike_end, width, height))

            spike_regions.append((spike_start, spike_end, width, height))

            in_spike = False


    return spike_regions


def find_and_process_txt_files(root_folder, output_folder, output_folder_csvs, dataset_type):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Walk through the directory
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            # Find files with .txt extension
            if file.endswith('.txt') and not file.endswith('.brkn.txt'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                plot_and_save_data(file_path, output_folder, output_folder_csvs, dataset_type)

# Example usage
root_folder = '/Users/ivanursul/Google Drive/My Drive/PhD/Dataset/Fall'  # Replace with your folder path
output_folder = os.path.join(root_folder, 'all_plots')  # All plots will be saved in this folder
output_folder_csvs = os.path.join(root_folder, 'all_csvs')  # All plots will be saved in this folder
find_and_process_txt_files(root_folder, output_folder, output_folder_csvs, 'Falls')

# Example usage
root_folder = '/Users/ivanursul/Google Drive/My Drive/PhD/Dataset/ADL'  # Replace with your folder path
output_folder = os.path.join(root_folder, 'all_plots')  # All plots will be saved in this folder
output_folder_csvs = os.path.join(root_folder, 'all_csvs')  # All plots will be saved in this folder
find_and_process_txt_files(root_folder, output_folder, output_folder_csvs, 'ADL')