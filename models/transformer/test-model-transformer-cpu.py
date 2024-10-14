import torch
import torch.nn as nn
import psutil
import os
import time

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

# Set the device to CPU as quantization is typically CPU-focused
device = torch.device('cpu')

# Initialize the model architecture first
input_dim = 4  # AccX, AccY, AccZ, etc.
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
