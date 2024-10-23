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
import optuna  # Import Optuna
import logging  # Import logging module
from training.dataset.fall_detection_dataset import FallDetectionDataset
from training.models.performer import PerformerModel
from training.models.t2v_bert import T2VBERTModel
from training.utils.dataset_utils import load_dataset
from training.utils.train_utils import train_model, evaluate_model, log_model_size, train_model_optuna
from training.utils.constants import fall_folder, non_fall_folder, max_sequence_length, input_dim, num_classes, \
    csv_columns

from training.utils.logging_utils import create_logger

# Set up logging
logger = create_logger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

file_paths, labels = load_dataset(fall_folder, non_fall_folder)

# Split data into training, validation, and test sets
train_files, temp_files, train_labels, temp_labels = train_test_split(file_paths, labels, test_size=0.3, random_state=42)
val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.5, random_state=42)

# Fit scaler on training data
scaler = StandardScaler()
all_train_data = []
for file in train_files:
    data = pd.read_csv(file)
    acc_data = data[['AccX_filtered_i', 'AccY_filtered_i', 'Corrected_AccZ_i', 'Acc_magnitude_i']].values
    all_train_data.append(acc_data)
all_train_data = np.vstack(all_train_data)
scaler.fit(all_train_data)

# Create PyTorch datasets
train_dataset = FallDetectionDataset(train_files, train_labels, scaler=scaler)
val_dataset = FallDetectionDataset(val_files, val_labels, scaler=scaler)
test_dataset = FallDetectionDataset(test_files, test_labels, scaler=scaler)


def objective(trial):
    # Predefine valid hidden_dim and num_heads combinations as tuples
    hidden_dim_num_heads_combinations = [
        (8, 1), (8, 2),
        (16, 2), (16, 4),
        (32, 2), (32, 4), (32, 8),
        (64, 2), (64, 4), (64, 8),
        (128, 2), (128, 4), (128, 8),
        (256, 4), (256, 8),
        (512, 8)
    ]

    # Optuna will now suggest a fixed combination of hidden_dim and num_heads
    hidden_dim, num_heads = trial.suggest_categorical('hidden_dim_num_heads', hidden_dim_num_heads_combinations)

    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Create DataLoaders with the current batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Build the model with hyperparameters
    model = T2VBERTModel(input_dim, num_heads, num_layers, num_classes, hidden_dim=hidden_dim, dropout=dropout).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    val_acc = train_model_optuna(device, criterion, model, optimizer, train_loader, trial, val_loader)

    return val_acc  # Use validation accuracy as the objective for Optuna


# Run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Retrieve the best hyperparameters
best_params = study.best_trial.params
logger.info(f"Best hyperparameters: {best_params}")

best_hidden_dim = best_params['hidden_dim']
best_num_heads = best_params['num_heads']
best_num_layers = best_params['num_layers']
best_dropout = best_params['dropout']
best_learning_rate = best_params['learning_rate']
best_batch_size = best_params['batch_size']

# Combine train and validation datasets
combined_train_files = train_files + val_files
combined_train_labels = train_labels + val_labels
combined_train_dataset = FallDetectionDataset(combined_train_files, combined_train_labels, scaler=scaler)

# Create DataLoaders with the best batch size
train_loader = DataLoader(combined_train_dataset, batch_size=best_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size)

# Build the model with the best hyperparameters
best_model = T2VBERTModel(input_dim, best_num_heads, best_num_layers, num_classes,
                          hidden_dim=best_hidden_dim, dropout=best_dropout).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_learning_rate)

# Retrain the model with best hyperparameters
num_epochs = 50  # Increase epochs for final training

train_model(device, best_model, train_loader, criterion, optimizer, num_epochs)

# Evaluate the best model
evaluate_model(best_model, test_loader)
