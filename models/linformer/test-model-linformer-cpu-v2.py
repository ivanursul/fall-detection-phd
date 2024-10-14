import torch
import torch.nn as nn
import psutil
import os
import time
from linformer import Linformer
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Define the Linformer-based Transformer model
class LinformerTransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim=128, dropout=0.011):
        super(LinformerTransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 800, hidden_dim))  # Adjust sequence length as needed

        # Using Linformer as a more efficient alternative to Transformer
        self.linformer = Linformer(
            dim=hidden_dim,
            seq_len=800,  # Fixed sequence length
            depth=num_layers,  # Number of transformer layers
            heads=num_heads,
            k=256,  # Low-rank approximation dimension
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.linformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(torch.__version__)

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

fall_folder = 'C:\\Users\\Ivan\\Downloads\\Sensor Data 6\\Fall'
non_fall_folder = 'C:\\Users\\Ivan\\Downloads\\Sensor Data 6\\ADL'

# Load file paths and labels
file_paths, labels = load_dataset(fall_folder, non_fall_folder)

# Initialize the Linformer model architecture
# input_dim = 4  # AccX, AccY, etc.
# num_heads = 4
# num_layers = 2
# num_classes = 2
# hidden_dim = 4
# dropout = 0.19302468696988134
max_sequence_length = 800

input_dim = 4  # AccX, AccY, AccZ
num_heads = 4
num_layers = 4
num_classes = 2  # Fall or non-fall
num_epochs = 20
dropout = 0.05346129216584067
hidden_dim = 128
learning_rate = 0.0007067641382961451

# Create the dataset and data loader
dataset = FallDetectionDataset(file_paths, labels, max_sequence_length, input_dim)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Create Linformer-based model
model = LinformerTransformerModel(input_dim, num_heads, num_layers, num_classes, hidden_dim, dropout).to(device)

# Load the model weights
model.load_state_dict(torch.load('linformer_transformer_model.pt', map_location=device))

# Ensure the model is in evaluation mode
model.eval()

# Track CPU usage during inference
start_time = time.time()
cpu_usage_before = psutil.cpu_percent(interval=None)  # Capture initial CPU usage

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

        predicted_class = torch.argmax(output, dim=1).item()

        # Measure CPU usage after inference
        cpu_usage_after = psutil.cpu_percent(interval=None)
        cpu_usage_during = cpu_usage_after - cpu_usage_before

        # Measure final memory usage
        memory_in_use_mb_after = process.memory_info().rss / (1024 ** 2)  # Memory in MB

        # Measure inference time
        inference_time = time.time() - start_time

        print(f"Classification Output: {predicted_class}, Actual Label: {batch_labels.item()}")
        print(f"RAM usage after inference: {memory_in_use_mb_after:.2f} MB")
        print(f"CPU usage during inference: {cpu_usage_during:.2f}%")
        print(f"Inference time: {inference_time:.4f} seconds")