import torch.nn as nn
import torch
from training.utils.constants import max_sequence_length


# Time2Vec layer
class Time2Vec(nn.Module):
    def __init__(self, input_dim):
        super(Time2Vec, self).__init__()
        self.w0 = nn.Parameter(torch.randn(1, input_dim))
        self.w = nn.Parameter(torch.randn(input_dim, input_dim))
        self.b = nn.Parameter(torch.randn(1, input_dim))
        self.b0 = nn.Parameter(torch.randn(1))

    def forward(self, x):
        v1 = torch.sin(torch.matmul(x, self.w) + self.b)
        v0 = self.w0 * x + self.b0
        return torch.cat([v0, v1], -1)

# T2V-BERT Model
class T2VBERTModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim, dropout):
        super(T2VBERTModel, self).__init__()
        self.time2vec = Time2Vec(input_dim)
        self.embedding = nn.Linear(input_dim * 2, hidden_dim)  # Time2Vec doubles the input size
        self.pos_encoder = nn.Parameter(torch.randn(1, max_sequence_length, hidden_dim))

        # Transformer encoder layers (like BERT)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.time2vec(x)  # Apply Time2Vec encoding
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x