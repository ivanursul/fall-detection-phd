import torch
import torch.nn as nn
import psutil
import os
import time
from linformer import Linformer

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

# Initialize the Linformer model architecture
input_dim = 4  # AccX, AccY, etc.
num_heads = 4
num_layers = 2
num_classes = 2
hidden_dim = 4
dropout = 0.19302468696988134

# Create Linformer-based model
model = LinformerTransformerModel(input_dim, num_heads, num_layers, num_classes, hidden_dim, dropout).to(device)

# Load the model weights
model.load_state_dict(torch.load('linformer_transformer_model.pt', map_location=device))

# Ensure the model is in evaluation mode
model.eval()

# Measure initial memory usage
process = psutil.Process(os.getpid())
memory_in_use_mb = process.memory_info().rss / (1024 ** 2)  # Memory in MB
print(f"Model loaded. Initial RAM usage: {memory_in_use_mb:.2f} MB")

# Generate dummy input to simulate a sequence with batch size of 32 and sequence length of 800
dummy_input = torch.randn(32, 800, input_dim).to(device)

# Track CPU usage during inference
start_time = time.time()
cpu_usage_before = psutil.cpu_percent(interval=None)  # Capture initial CPU usage

# Perform inference
with torch.no_grad():
    output = model(dummy_input)

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
