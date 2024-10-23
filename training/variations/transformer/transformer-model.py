from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from training.models.transformer import TransformerModel
from training.dataset.fall_detection_dataset import FallDetectionDataset
from training.utils.dataset_utils import load_dataset
from training.utils.train_utils import train_model, evaluate_model
from training.utils.constants import fall_folder, non_fall_folder, input_dim, num_classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

print(torch.__version__)

# Constants
BATCH_SIZE = 32

num_heads = 4
num_layers = 2
num_epochs = 20
dropout = 0.011037393228528439
hidden_dim=128
learning_rate = 0.0005798324832380791

file_paths, labels = load_dataset(fall_folder, non_fall_folder)

# Split data into training and test sets
train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

# Create PyTorch datasets and loaders
train_dataset = FallDetectionDataset(train_files, train_labels)
test_dataset = FallDetectionDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = TransformerModel(input_dim, num_heads, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, train_loader, test_loader, criterion, optimizer, epochs=num_epochs)

torch.save(model.state_dict(), 'transformer_model.pt')

# Evaluate the model on the test set
evaluate_model(model, test_loader)



