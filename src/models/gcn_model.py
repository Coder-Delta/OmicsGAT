from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNModel(nn.Module):
    """Two-layer graph-level GCN classifier; API matches ``GATModel``."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, heads: int = 1, dropout: float = 0.2):
        super().__init__()
        del heads  # accepted for drop-in compatibility with GATModel
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.classifier(global_mean_pool(x, batch))
