import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from training.dataset.fall_detection_dataset import FallDetectionDataset
from training.models.multi_scale_transformer import MultiScaleTransformerModel
from training.utils.dataset_utils import load_dataset
from training.utils.train_utils import train_model, evaluate_model
from training.utils.constants import fall_folder, non_fall_folder, max_sequence_length, input_dim, num_classes


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Constants
BATCH_SIZE = 16
num_heads = 8
num_layers = 2
num_epochs = 50
dropout = 0.2
hidden_dim = 48
learning_rate = 0.00034872838768668134

file_paths, labels = load_dataset(fall_folder, non_fall_folder)

# Split data into training and test sets
train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2,
                                                                      random_state=42)

# Create PyTorch datasets and loaders
train_dataset = FallDetectionDataset(train_files, train_labels)
test_dataset = FallDetectionDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = MultiScaleTransformerModel(input_dim, num_heads, num_layers, num_classes,
                                   hidden_dim=hidden_dim, dropout=dropout).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(device, model, train_loader, test_loader, criterion, optimizer, epochs=num_epochs)


# Save the trained model
model_save_path = 'multi-scalel-transformer-model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# Evaluate the model on the test set
evaluate_model(device, model, test_loader)
