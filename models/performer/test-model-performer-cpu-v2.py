import torch
import torch.nn as nn
import psutil
import os
import time
from performer_pytorch import Performer
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

MAX_SEQUENCE_LENGTH = 800
BATCH_SIZE = 32
input_dim = 4  # AccX, AccY, AccZ
num_heads = 4
num_layers = 2
num_classes = 2  # Fall or non-fall
num_epochs = 20
dropout = 0.011037393228528439
hidden_dim = 128
learning_rate = 0.0005798324832380791

# Modified TransformerModel to use Performer
class PerformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim=hidden_dim, dropout=dropout):
        super(PerformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_SEQUENCE_LENGTH, hidden_dim))

        # Use Performer instead of the regular Transformer encoder
        self.performer = Performer(
            dim=hidden_dim,          # Hidden dimension
            depth=num_layers,        # Number of layers
            heads=num_heads,         # Number of attention heads
            dim_head=hidden_dim // num_heads,  # Dimension of each attention head
            causal=False,            # Set to True if you want causal self-attention
            ff_dropout=dropout,      # Feed-forward dropout
            attn_dropout=dropout     # Attention dropout
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.performer(x)
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
input_dim = 4  # AccX, AccY, etc.
num_heads = 4
num_layers = 2
num_classes = 2
hidden_dim = 4
dropout = 0.19302468696988134
max_sequence_length = 800

# Create the dataset and data loader
dataset = FallDetectionDataset(file_paths, labels, max_sequence_length, input_dim)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize Performer model
model = PerformerModel(input_dim, num_heads, num_layers, num_classes).to(device)

# Load the model weights
model.load_state_dict(torch.load('performer_model.pt', map_location=device))

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