fall_folder = 'C:\\Users\\Ivan\\Downloads\\Sensor Data 6\\Fall'
non_fall_folder = 'C:\\Users\\Ivan\\Downloads\\Sensor Data 6\\ADL'

max_sequence_length = 800  # Adjust based on your dataset
input_dim = 4  # AccX, AccY, AccZ
num_classes = 2
csv_columns = ['AccX_filtered_i', 'AccY_filtered_i', 'Corrected_AccZ_i', 'Acc_magnitude_i']