import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from linformer import Linformer  # Import Linformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(f'Using device: {device}')

# Constants
MAX_SEQUENCE_LENGTH = 800
BATCH_SIZE = 16

input_dim = 4  # AccX, AccY, AccZ
num_heads = 4
num_layers = 1
num_classes = 2  # Fall or non-fall
num_epochs = 50
dropout = 0.5
hidden_dim = 128
learning_rate = 0.0006939418263593155


# Custom Dataset class (same as your original code)
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

        acc_data_scaled = self.scaler.fit_transform(acc_data)

        add_data_final = acc_data_scaled

        if len(add_data_final) < MAX_SEQUENCE_LENGTH:
            padded = np.zeros((MAX_SEQUENCE_LENGTH, input_dim))
            padded[:len(add_data_final), :] = add_data_final
        else:
            padded = add_data_final[:MAX_SEQUENCE_LENGTH, :]

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

train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2,
                                                                      random_state=42)

train_dataset = FallDetectionDataset(train_files, train_labels)
test_dataset = FallDetectionDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# Linformer-based Transformer Model
class LinformerTransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim=hidden_dim, dropout=dropout):
        super(LinformerTransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_SEQUENCE_LENGTH, hidden_dim))

        # Linformer: Use Linformer as the attention mechanism
        self.linformer = Linformer(
            dim=hidden_dim,  # Hidden size (same as d_model in standard transformer)
            seq_len=MAX_SEQUENCE_LENGTH,  # Sequence length
            depth=num_layers,  # Number of layers
            heads=num_heads,  # Number of attention heads
            k=256,  # Compression factor for low-rank approximation
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.linformer(x)  # Pass through Linformer
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x


model = LinformerTransformerModel(input_dim, num_heads, num_layers, num_classes).to(device)

# Loss and optimizer (same as original code)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop (same as original code)
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

torch.save(model.state_dict(), 'linformer_transformer_model.pt')


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Fall', 'Fall']))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Fall', 'Fall'],
                yticklabels=['Non-Fall', 'Fall'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
# Evaluate the model
evaluate_model(model, test_loader)

