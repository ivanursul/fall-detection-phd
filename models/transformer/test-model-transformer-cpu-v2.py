import torch
import torch.nn as nn
import psutil
import os
import time
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np


# Define the same Transformer model structure for deployment
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim=64, dropout=0.011037393228528439):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 800, hidden_dim))  # Adjust sequence length as needed
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x


# Custom Dataset class
class FallDetectionDataset(Dataset):
    def __init__(self, file_paths, labels, max_sequence_length, input_dim):
        self.file_paths = file_paths
        self.labels = labels
        self.scaler = StandardScaler()
        self.max_sequence_length = max_sequence_length
        self.input_dim = input_dim

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = pd.read_csv(self.file_paths[idx])
        acc_data = data[['AccX_filtered_i', 'AccY_filtered_i', 'Corrected_AccZ_i', 'Acc_magnitude_i']].values
        acc_data = self.scaler.fit_transform(acc_data)

        # Padding or truncating to match sequence length
        if len(acc_data) < self.max_sequence_length:
            padded = np.zeros((self.max_sequence_length, self.input_dim))
            padded[:len(acc_data), :] = acc_data
        else:
            padded = acc_data[:self.max_sequence_length, :]

        tensor_data = torch.tensor(padded, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_data, label


# Load file paths and labels
def load_dataset(fall_folder, non_fall_folder):
    def filter_files(folder):
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

    fall_files = filter_files(fall_folder)
    non_fall_files = filter_files(non_fall_folder)

    file_paths = fall_files + non_fall_files
    labels = [1] * len(fall_files) + [0] * len(non_fall_files)

    return file_paths, labels


# Set directories for Fall and ADL (non-fall) activities
fall_folder = 'C:\\Users\\Ivan\\Downloads\\Sensor Data 6\\Fall'
non_fall_folder = 'C:\\Users\\Ivan\\Downloads\\Sensor Data 6\\ADL'

# Load file paths and labels
file_paths, labels = load_dataset(fall_folder, non_fall_folder)

# Set parameters
input_dim = 4  # AccX, AccY, Corrected_AccZ, Acc_magnitude
max_sequence_length = 800
batch_size = 32

# Create the dataset and data loader
dataset = FallDetectionDataset(file_paths, labels, max_sequence_length, input_dim)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set the device to CPU as quantization is typically CPU-focused
device = torch.device('cpu')

# Initialize the model architecture
num_heads = 4
num_layers = 2
num_classes = 2
hidden_dim = 128
dropout = 0.011037393228528439

model = TransformerModel(input_dim, num_heads, num_layers, num_classes, hidden_dim, dropout).to(device)

# Load the state_dict into the model
model.load_state_dict(torch.load('transformer_model.pt', map_location=device))

# Ensure the model is in evaluation mode
model.eval()

# Measure initial memory usage
process = psutil.Process(os.getpid())
memory_in_use_mb = process.memory_info().rss / (1024 ** 2)  # Memory in MB
print(f"Model loaded. Initial RAM usage: {memory_in_use_mb:.2f} MB")


with torch.no_grad():
    for batch_data, batch_labels in data_loader:
        # Perform inference on the data from the data loader
        start_time = time.time()
        cpu_usage_before = psutil.cpu_percent(interval=None)

        batch_data = batch_data.to(device)
        output = model(batch_data)

        # Print the output for each batch
        #print(f"Output for batch: {output}")
        #print(f"Predicted classes: {torch.argmax(output, dim=1)}")  # Predicted class for each sample

        # Measure CPU usage after inference
        cpu_usage_after = psutil.cpu_percent(interval=None)
        cpu_usage_during = cpu_usage_after - cpu_usage_before

        # Measure final memory usage
        memory_in_use_mb_after = process.memory_info().rss / (1024 ** 2)  # Memory in MB

        # Measure inference time
        inference_time = time.time() - start_time

        print(f"RAM usage after inference: {memory_in_use_mb_after:.2f} MB")
        print(f"CPU usage during inference: {cpu_usage_during:.2f}%")
        print(f"Inference time: {inference_time:.4f} seconds")


