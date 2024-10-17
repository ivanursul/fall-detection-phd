import time
import math
import os
import pandas as pd
import smbus
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
from scipy.signal import butter, filtfilt
import RPi.GPIO as GPIO
from collections import deque


# Define GPIO pin for the buzzer
buzzer_pin = 23

# Set up GPIO mode (only done once, no need to clean up after each call)
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer_pin, GPIO.OUT)

# MPU6050 Registers and their Addresses
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F

# BMP388 I2C address
BMP388_ADDRESS = 0x77  # Adjust based on your device address
# BMP388 Register for pressure
BMP388_REG_PRESS_MSB = 0x04  # Pressure MSB register

bus = smbus.SMBus(1)  # I2C bus number on Raspberry Pi

# Set the sampling interval for 100Hz (0.01 seconds per sample)
sampling_interval = 0.01
# Use a deque to hold the last 100 altitude readings for moving average
window_size = 100  # Moving average over 1 second (100 samples)
altitude_window = deque(maxlen=window_size)


def read_raw_data(addr):
    # Reads the raw data from the specified address (high byte and low byte)
    high = bus.read_byte_data(MPU6050_ADDR, addr)
    low = bus.read_byte_data(MPU6050_ADDR, addr + 1)
    # Combine high and low bytes
    value = (high << 8) | low
    # Convert to signed value if needed
    if value > 32768:
        value = value - 65536
    return value


# Function to read raw pressure data from BMP388
def read_bmp388_pressure():
    # Read 3 bytes of pressure data (MSB, LSB, XLSB)
    press_msb = bus.read_byte_data(BMP388_ADDRESS, BMP388_REG_PRESS_MSB)
    press_lsb = bus.read_byte_data(BMP388_ADDRESS, BMP388_REG_PRESS_MSB + 1)
    press_xlsb = bus.read_byte_data(BMP388_ADDRESS, BMP388_REG_PRESS_MSB + 2)

    # Combine pressure data
    pressure_raw = (press_msb << 16) | (press_lsb << 8) | press_xlsb
    return pressure_raw


def calculate_altitude(pressure, sea_level_pressure=1013.25):
    """
    Calculate altitude from pressure using the barometric formula.
    :param pressure: Current pressure in hPa
    :param sea_level_pressure: Sea level standard atmospheric pressure in hPa (default: 1013.25)
    :return: Altitude in meters
    """
    altitude = 44330.0 * (1.0 - (pressure / sea_level_pressure) ** 0.1903)
    return altitude


def initialize_mpu(accel_range):
    # Wake up the MPU6050 as it starts in sleep mode
    bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0x00)

    # Set the accelerometer range
    if accel_range == 2:
        accel_config_value = 0x00  # ±2g
        scaling_factor = 16384.0
    elif accel_range == 4:
        accel_config_value = 0x08  # ±4g
        scaling_factor = 8192.0
    elif accel_range == 8:
        accel_config_value = 0x10  # ±8g
        scaling_factor = 4096.0
    elif accel_range == 16:
        accel_config_value = 0x18  # ±16g
        scaling_factor = 2048.0
    else:
        raise ValueError("Invalid accelerometer range. Choose from 2, 4, 8, 16.")

    bus.write_byte_data(MPU6050_ADDR, ACCEL_CONFIG, accel_config_value)

    return scaling_factor


def load_offsets(filename):
    with open(filename, 'r') as f:
        offsets = json.load(f)
    print(f"Offsets loaded from {filename}")
    return offsets["acc_x_offset"], offsets["acc_y_offset"], offsets["acc_z_offset"]


acc_x_offset, acc_y_offset, acc_z_offset = load_offsets("/home/ivanursul/mpu_offsets.json")

def read_accelerometer(scaling_factor):
    # Reading the raw accelerometer values
    acc_x = read_raw_data(ACCEL_XOUT_H) - acc_x_offset
    acc_y = read_raw_data(ACCEL_YOUT_H) - acc_y_offset
    acc_z = read_raw_data(ACCEL_ZOUT_H) - acc_z_offset

    # Scale the raw accelerometer values to g-units based on the selected range
    acc_x_g = acc_x / scaling_factor
    acc_y_g = acc_y / scaling_factor
    acc_z_g = acc_z / scaling_factor

    return acc_x_g, acc_y_g, acc_z_g


# Function to calculate the magnitude of a vector (x, y, z)
def calculate_magnitude(x, y, z):
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)


accel_range = 16
scaling_factor = initialize_mpu(accel_range)

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

            spike_regions.append((spike_start, spike_end, width, height))

            in_spike = False

    return spike_regions


def apply_linear_interpolation_for_spikes(df, spike_regions, acc_x_column, acc_y_column, acc_z_column):
    # Remove the spike regions and interpolate the values for AccX, AccY, and AccZ
    for spike_start, spike_end, width, height in spike_regions:
        # Set the spike region to NaN for all AccX(g), AccY(g), and AccZ(g)
        df.loc[spike_start:spike_end, [acc_x_column, acc_y_column, acc_z_column]] = np.nan

        # Perform linear interpolation for AccX(g), AccY(g), and AccZ(g)
        df[acc_x_column] = df[acc_x_column].interpolate(method='linear')
        df[acc_y_column] = df[acc_y_column].interpolate(method='linear')
        df[acc_z_column] = df[acc_z_column].interpolate(method='linear')

    # Fill any remaining NaN values using forward or backward filling (no inplace)
    df.fillna({acc_x_column: df[acc_x_column].ffill(),
               acc_y_column: df[acc_y_column].ffill(),
               acc_z_column: df[acc_z_column].bfill()}, inplace=True)

    return df


def detect_multiple_spikes(df, window_size=10, density_threshold=0.5):
    # Calculate a rolling window to detect dense segments of high magnitude
    df['rolling_sum'] = df['Magnitude'].rolling(window=window_size).sum()

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


def expand_fall_spike_for_multiple_spikes(df, spikes, threshold_factor=0.3):
    expanded_spikes = []

    for (fall_start, fall_end) in spikes:
        # Get the peak value within the current spike
        peak_value = df['Magnitude'].iloc[fall_start:fall_end].max()
        min_threshold = peak_value * threshold_factor

        # Extend backwards
        while fall_start > 0 and df['Magnitude'].iloc[fall_start] > min_threshold:
            fall_start -= 1

        # Extend forwards
        while fall_end < len(df) - 1 and df['Magnitude'].iloc[fall_end] > min_threshold:
            fall_end += 1

        # Add the expanded spike to the list
        expanded_spikes.append((fall_start, fall_end))

    return expanded_spikes


def apply_butter_lowpass_filter_for_non_fall_segments(df, spikes, cutoff=5, fs=100, order=4):
    # Initialize filtered columns with original data
    df['acc_x_filtered'] = df['AccX']
    df['acc_y_filtered'] = df['AccY']
    df['acc_z_filtered'] = df['AccZ']

    # Get all non-spike regions
    last_end = 0
    for spike_start, spike_end in spikes:
        # Filter data before the spike
        if last_end < spike_start:
            df.loc[last_end:spike_start - 1, 'acc_x_filtered'] = butter_lowpass_filter(
                df['AccX'].iloc[last_end:spike_start], cutoff, fs, order)
            df.loc[last_end:spike_start - 1, 'acc_y_filtered'] = butter_lowpass_filter(
                df['AccY'].iloc[last_end:spike_start], cutoff, fs, order)
            df.loc[last_end:spike_start - 1, 'acc_z_filtered'] = butter_lowpass_filter(
                df['AccZ'].iloc[last_end:spike_start], cutoff, fs, order)

        # After each spike, set the last end to the spike end
        last_end = spike_end + 1

    # Filter the remaining data after the last spike
    if last_end < len(df):
        df.loc[last_end:, 'acc_x_filtered'] = butter_lowpass_filter(df['AccX'].iloc[last_end:], cutoff, fs, order)
        df.loc[last_end:, 'acc_y_filtered'] = butter_lowpass_filter(df['AccY'].iloc[last_end:], cutoff, fs, order)
        df.loc[last_end:, 'acc_z_filtered'] = butter_lowpass_filter(df['AccZ'].iloc[last_end:], cutoff, fs, order)

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

def filter_data(df):
    cutoff = 5
    fs = 100
    window_size = 10
    fall_width_threshold = 27
    height_threshold = 2.0

    df['Magnitude'] = np.sqrt(df['AccX'] ** 2 + df['AccY'] ** 2 + df['AccZ'] ** 2)

    # Calculate spikes for non-filtered data
    spike_regions_raw_data = calculate_all_spikes(df, height_threshold, 'Magnitude')

    # Find all spikes that exceed certain width and height
    filtered_spike_regions = [
        (spike_start, spike_end, width, height)
        for spike_start, spike_end, width, height in spike_regions_raw_data
        if width <= 2 and height >= 2.0
    ]

    # Smooth spikes out
    apply_linear_interpolation_for_spikes(df, filtered_spike_regions, 'AccX', 'AccY', 'AccZ')

    fall_spikes = detect_multiple_spikes(df, window_size, fall_width_threshold)
    fall_spikes = expand_fall_spike_for_multiple_spikes(df, fall_spikes, threshold_factor=0.3)
    df = apply_butter_lowpass_filter_for_non_fall_segments(df, fall_spikes, cutoff, fs)

    df['acc_z_filtered'] = df['acc_z_filtered'] - 1

    # Calculate the magnitude of the accelerometer (AccX, AccY, AccZ)
    df['acc_magnitude_filtered'] = np.sqrt(
        df['acc_x_filtered'] ** 2 + df['acc_y_filtered'] ** 2 + df['acc_z_filtered'] ** 2
    )

    return df


scaler = StandardScaler()


def play_alarm(beep_count=5, beep_duration=0.2, pause_duration=0.1):
    """
    Play a repetitive beep alarm for fall detection.

    Parameters:
    beep_count (int): Number of beeps in the alarm sequence.
    beep_duration (float): Duration of each beep in seconds.
    pause_duration (float): Pause between each beep in seconds.
    """
    for _ in range(beep_count):
        p = GPIO.PWM(buzzer_pin, 1000)  # 1000Hz frequency for the alarm tone
        p.start(50)  # 50% duty cycle for consistent sound
        time.sleep(beep_duration)  # Duration of the beep
        p.stop()  # Stop the PWM after the beep
        time.sleep(pause_duration)  # Short pause between beeps


def signal_app_start(beep_count=3, beep_duration=0.1, pause_duration=0.05):
    """
    Signal the start of the application by playing a distinct beep sequence.

    Parameters:
    beep_count (int): Number of beeps in the start signal sequence.
    beep_duration (float): Duration of each beep in seconds.
    pause_duration (float): Pause between each beep in seconds.
    """
    for _ in range(beep_count):
        p = GPIO.PWM(buzzer_pin, 2000)  # 2000Hz frequency for start signal
        p.start(50)  # 50% duty cycle for consistent sound
        time.sleep(beep_duration)  # Duration of the beep
        p.stop()  # Stop the PWM after the beep
        time.sleep(pause_duration)  # Short pause between beeps


def collect():
    play_alarm(beep_count=1, beep_duration=0.2, pause_duration=0.2)
    data_records = collect_interval_records()

    filtered_data = filter_data(data_records)

    save_csv(filtered_data)

    start_time = time.time()

    end_time = time.time()
    collection_time = end_time - start_time
    print(f"Time to check if there was a fall: {collection_time:.4f} seconds")

    play_alarm(beep_count=2, beep_duration=0.2, pause_duration=0.2)


def save_csv(filtered_data):
    # Ensure the folder exists
    folder_path = '/home/ivanursul/accelerometer_data_raw'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Generate a unique filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"accelerometer_data_{timestamp}.csv"
    file_path = os.path.join(folder_path, filename)
    # Convert the data_records to a DataFrame for saving to CSV
    filtered_data.to_csv(file_path, index=False)


def collect_interval_records():
    data_records = pd.DataFrame(
        0.0, index=np.arange(800),
        columns=['AccX', 'AccY', 'AccZ', 'Magnitude', 'MovAvgAltitude']
    )

    # Collect 800 records at 100Hz
    for i in range(800):
        # Read the sensor data
        acc_x, acc_y, acc_z = read_accelerometer(scaling_factor)

        pressure_raw = read_bmp388_pressure()
        current_altitude = calculate_altitude(pressure_raw)

        # Append the current altitude to the moving window
        altitude_window.append(current_altitude)

        # Calculate the moving average if we have enough data points
        if len(altitude_window) == window_size:
            moving_avg_altitude = sum(altitude_window) / len(altitude_window)
            print(f"Moving Average Altitude: {moving_avg_altitude:.2f} meters")

        # Calculate magnitude
        magnitude = calculate_magnitude(acc_x, acc_y, acc_z - 1, )

        # Assign the data to the DataFrame
        data_records.iloc[i] = [acc_x, acc_y, acc_z - 1, magnitude]

        # Wait for 10ms (100Hz frequency)
        time.sleep(0.01)
    return data_records


signal_app_start()
try:
    while True:
        collect()
finally:
    GPIO.cleanup()