import torch.nn as nn
import torch
from performer_pytorch import Performer
from training.utils.constants import max_sequence_length

# Modified TransformerModel to use Performer
class PerformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim, dropout):
        super(PerformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_sequence_length, hidden_dim))

        # Use Performer instead of the regular Transformer encoder
        self.performer = Performer(
            dim=hidden_dim,
            depth=num_layers,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            causal=False,
            ff_dropout=dropout,
            attn_dropout=dropout,
            nb_features=64  # Add this line to ensure nb_features is not zero
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.performer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x