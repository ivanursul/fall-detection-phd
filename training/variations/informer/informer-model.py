from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from training.dataset.fall_detection_dataset import FallDetectionDataset
from training.models.informer import InformerClassificationModel
from training.utils.dataset_utils import load_dataset
from training.utils.train_utils import train_model, evaluate_model
from training.utils.constants import fall_folder, non_fall_folder, max_sequence_length, input_dim, num_classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
d_model = 128
batch_size = 16
num_heads = 4
num_layers = 1
num_epochs = 50
dropout = 0.1
learning_rate = 0.0005756017680021591

file_paths, labels = load_dataset(fall_folder, non_fall_folder)

# Split data into training and test sets
train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

# Create PyTorch datasets and loaders
train_dataset = FallDetectionDataset(train_files, train_labels)
test_dataset = FallDetectionDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = InformerClassificationModel(
        input_dim=input_dim,
        seq_len=max_sequence_length,
        num_classes=num_classes,
        d_model=d_model,
        n_heads=num_heads,
        e_layers=num_layers,
        dropout=dropout
    ).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(device, model, train_loader, test_loader, criterion, optimizer, epochs=num_epochs)

# Save the trained model
model_save_path = 'informer.py-memory-model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# Evaluate the model on the test set
evaluate_model(device, model, test_loader)
