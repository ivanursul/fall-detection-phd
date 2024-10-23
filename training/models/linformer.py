import torch.nn as nn
import torch
from linformer import Linformer


# Linformer-based Transformer Model
class LinformerTransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim, dropout, max_sequence_length):
        super(LinformerTransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_sequence_length, hidden_dim))

        # Linformer: Use Linformer as the attention mechanism
        self.linformer = Linformer(
            dim=hidden_dim,  # Hidden size (same as d_model in standard transformer)
            seq_len=max_sequence_length,  # Sequence length
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