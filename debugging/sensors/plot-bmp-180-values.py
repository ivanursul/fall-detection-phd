import os
import pandas as pd
import matplotlib.pyplot as plt

# Define directories
DATA_DIRECTORY = '/Users/ivanursul/Downloads/sensors/sensor_data'
OUTPUT_DIRECTORY = '/Users/ivanursul/Downloads/sensors/plots'

MOVING_AVERAGE_WINDOW = 50  # Number of records to calculate the moving average

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
    print(f"Created directory '{OUTPUT_DIRECTORY}' for storing output charts.")

# Function to generate charts for pressure and altitude from a single CSV
def generate_chart_for_csv(filepath, output_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filepath)

    # Ensure the DataFrame contains the necessary columns
    if 'pressure' in df.columns and 'altitude' in df.columns and 'timestamp' in df.columns:
        # Calculate the moving average for pressure and altitude
        df['pressure_moving_avg'] = df['pressure'].rolling(window=MOVING_AVERAGE_WINDOW).mean()
        df['altitude_moving_avg'] = df['altitude'].rolling(window=MOVING_AVERAGE_WINDOW).mean()

        # Create a figure with two subplots (one for pressure, one for altitude)
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot Pressure vs Time with moving average
        axs[0].plot(df['timestamp'], df['pressure'], label='Pressure (Pa)', color='blue', alpha=0.3)
        axs[0].plot(df['timestamp'], df['pressure_moving_avg'], label=f'Moving Avg (n={MOVING_AVERAGE_WINDOW})', color='red')
        axs[0].set_xlabel('Timestamp')
        axs[0].set_ylabel('Pressure (Pa)')
        axs[0].set_title('Pressure Over Time with Moving Average')
        axs[0].legend()

        # Plot Altitude vs Time with moving average
        axs[1].plot(df['timestamp'], df['altitude'], label='Altitude (meters)', color='green', alpha=0.3)
        axs[1].plot(df['timestamp'], df['altitude_moving_avg'], label=f'Moving Avg (n={MOVING_AVERAGE_WINDOW})', color='red')
        axs[1].set_xlabel('Timestamp')
        axs[1].set_ylabel('Altitude (meters)')
        axs[1].set_title('Altitude Over Time with Moving Average')
        axs[1].legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure as a PNG file in the output directory
        plt.savefig(output_path)
        plt.close()

        print(f"Chart saved as {output_path}")
    else:
        print(f"Required columns 'pressure' and 'altitude' not found in {filepath}.")

# Main function to process all CSV files
def main():
    # Loop through all CSV files in the data directory
    for filename in os.listdir(DATA_DIRECTORY):
        if filename.endswith('.csv'):
            filepath = os.path.join(DATA_DIRECTORY, filename)
            output_filepath = os.path.join(OUTPUT_DIRECTORY, f"{os.path.splitext(filename)[0]}_chart.png")
            print(f"Processing file: {filename}")
            # Generate the chart for the current CSV file
            generate_chart_for_csv(filepath, output_filepath)

if __name__ == "__main__":
    main()
