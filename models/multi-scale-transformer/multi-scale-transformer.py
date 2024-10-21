import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Constants
MAX_SEQUENCE_LENGTH = 800  # Adjust based on your dataset
BATCH_SIZE = 16
input_dim = 4  # AccX, AccY, AccZ
num_heads = 8
num_layers = 2
num_classes = 2  # Fall or non-fall
num_epochs = 50
dropout = 0.2
hidden_dim = 48
learning_rate = 0.00034872838768668134

# Custom Dataset class
class FallDetectionDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.scaler = StandardScaler()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = pd.read_csv(self.file_paths[idx])
        acc_data = data[['AccX_filtered_i', 'AccY_filtered_i', 'Corrected_AccZ_i', 'Acc_magnitude_i']].values
        acc_data = self.scaler.fit_transform(acc_data)

        if len(acc_data) < MAX_SEQUENCE_LENGTH:
            padded = np.zeros((MAX_SEQUENCE_LENGTH, input_dim))
            padded[:len(acc_data), :] = acc_data
        else:
            padded = acc_data[:MAX_SEQUENCE_LENGTH, :]

        tensor_data = torch.tensor(padded, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_data, label


# Load file paths and labels
def load_dataset(fall_folder, non_fall_folder):
    def filter_files(folder):
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv') or f.endswith('.txt')]

    fall_files = filter_files(fall_folder)
    non_fall_files = filter_files(non_fall_folder)

    file_paths = fall_files + non_fall_files
    labels = [1] * len(fall_files) + [0] * len(non_fall_files)

    return file_paths, labels


fall_folder = 'C:\\Users\\Ivan\\Downloads\\Sensor Data 6\\Fall'
non_fall_folder = 'C:\\Users\\Ivan\\Downloads\\Sensor Data 6\\ADL'

file_paths, labels = load_dataset(fall_folder, non_fall_folder)

# Split data into training and test sets
train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2,
                                                                      random_state=42)

# Create PyTorch datasets and loaders
train_dataset = FallDetectionDataset(train_files, train_labels)
test_dataset = FallDetectionDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQUENCE_LENGTH):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: batch_size x seq_len x d_model
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQUENCE_LENGTH):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: batch_size x seq_len x d_model
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

# Multi-Scale Transformer Model
class MultiScaleTransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim=128, dropout=0.1):
        super(MultiScaleTransformerModel, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Define multiple scales
        self.scale_factors = [1, 2, 4]  # Adjust scales as needed

        self.transformers = nn.ModuleList()
        self.position_encodings = nn.ModuleList()

        for scale in self.scale_factors:
            # Create transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
            transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.transformers.append(transformer)

            # Positional encoding
            max_len = int(np.ceil(MAX_SEQUENCE_LENGTH / scale))
            pe = PositionalEncoding(hidden_dim, max_len=max_len)
            self.position_encodings.append(pe)

        # Linear layer to match the transformer d_model
        self.linear = nn.Linear(input_dim, hidden_dim)

        # Classification layer
        self.fc = nn.Linear(hidden_dim * len(self.scale_factors), num_classes)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: batch_size x seq_len x input_dim
        outputs = []
        for idx, scale in enumerate(self.scale_factors):
            # Downsample the sequence
            if scale > 1:
                x_scaled = x[:, ::scale, :]
            else:
                x_scaled = x
            # Linear projection
            x_scaled = self.linear(x_scaled)
            # Add positional encoding
            x_scaled = self.position_encodings[idx](x_scaled)
            # Pass through transformer
            x_scaled = self.transformers[idx](x_scaled)
            # Pooling (e.g., global average pooling)
            x_pooled = x_scaled.mean(dim=1)
            outputs.append(x_pooled)

        # Concatenate outputs from different scales
        x_concat = torch.cat(outputs, dim=1)  # Shape: batch_size x (hidden_dim * num_scales)
        x_concat = self.dropout_layer(self.relu(x_concat))
        # Classification layer
        x_out = self.fc(x_concat)
        return x_out

model = MultiScaleTransformerModel(input_dim, num_heads, num_layers, num_classes,
                                   hidden_dim=hidden_dim, dropout=dropout).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train model function
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=num_epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%')

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_acc = 100 * correct / total
        print(f'Test Accuracy: {test_acc:.2f}%')


# Train the model
train_model(model, train_loader, test_loader, criterion, optimizer, epochs=num_epochs)


# Save the trained model
model_save_path = 'multi-scalel-transformer-model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')


# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Fall', 'Fall']))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Fall', 'Fall'],
                yticklabels=['Non-Fall', 'Fall'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# Evaluate the model on the test set
evaluate_model(model, test_loader)
