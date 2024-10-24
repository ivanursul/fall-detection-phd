
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import optuna

from training.dataset.fall_detection_dataset import FallDetectionDataset

from training.models.transformer import TransformerModel
from training.utils.dataset_utils import load_dataset
from training.utils.train_utils import train_model, evaluate_model, log_model_size
from training.utils.constants import fall_folder, non_fall_folder, max_sequence_length, input_dim, num_classes, \
    csv_columns


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Constants
num_epochs = 10  # Reduced to speed up the hyperparameter search
num_trials = 500

file_paths, labels = load_dataset(fall_folder, non_fall_folder)
train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

train_dataset = FallDetectionDataset(train_files, train_labels)
test_dataset = FallDetectionDataset(test_files, test_labels)


# Function to create the model with specific hyperparameters
def create_model(input_dim, num_heads, num_layers, hidden_dim, dropout, learning_rate):
    model = TransformerModel(input_dim=input_dim, num_heads=num_heads, num_layers=num_layers,
                             num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer



valid_combinations = [
    (4, 4),(4, 8), (4, 16),(4, 32),(4, 64),(4, 128),
]


def objective(trial):
    # Select valid combination of num_heads and hidden_dim
    num_heads, hidden_dim = trial.suggest_categorical('num_heads_hidden_dim', valid_combinations)

    # Tune other hyperparameters
    num_layers = trial.suggest_int('num_layers', 2, 4)  # Number of transformer layers
    dropout = trial.suggest_float('dropout', 0.01, 0.1)  # Dropout rate
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)  # Learning rate
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])  # Batch size

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create model and optimizer

    model, optimizer = create_model(input_dim=input_dim, num_heads=num_heads, num_layers=num_layers,
                                    hidden_dim=hidden_dim, dropout=dropout, learning_rate=learning_rate)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Train model
    train_model(device, model, train_loader, test_loader, criterion, optimizer, epochs=num_epochs)

    # Evaluate model
    accuracy = evaluate_model(model, test_loader, return_acc=True)
    return accuracy


# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=num_trials)

# Best hyperparameters found by Optuna
print(f'Best trial: {study.best_trial.params}')

# Evaluate the best model with the tuned hyperparameters
best_params = study.best_trial.params
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])

best_model, best_optimizer = create_model(input_dim=input_dim, num_heads=best_params['num_heads'], num_layers=best_params['num_layers'],
                                          hidden_dim=best_params['hidden_dim'], dropout=best_params['dropout'], learning_rate=best_params['learning_rate'])
best_model = best_model.to(device)

train_model(device, best_model, train_loader, nn.CrossEntropyLoss(), best_optimizer, epochs=num_epochs)
evaluate_model(best_model, test_loader)
