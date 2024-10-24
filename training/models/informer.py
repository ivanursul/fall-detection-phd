import torch.nn as nn
import torch
import numpy as np

def get_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)
    return pe


class InformerClassificationModel(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes, d_model=128, n_heads=4, e_layers=2, dropout=0.1):
        super(InformerClassificationModel, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model

        # Embedding layers
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = get_positional_encoding(seq_len, d_model)

        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Classification head
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        # Embed input
        x = self.value_embedding(x)  # Shape: (batch_size, seq_len, d_model)

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)

        # Permute for transformer input: (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)

        # Pass through encoder
        x = self.encoder(x)  # Shape: (seq_len, batch_size, d_model)

        # Global average pooling
        x = x.mean(dim=0)  # Shape: (batch_size, d_model)

        # Classification layer
        x = self.fc(x)  # Shape: (batch_size, num_classes)

        return x
