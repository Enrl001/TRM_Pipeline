"""Graph attention network on the per-window connectivity graphs (model 2).

One graph per window: nodes = EEG channels, node features = log band power
(optionally plus the node's connectivity profile), edges = the connectivity graph
after the same filtering as the baseline (graph_metrics.filter_graph), with the
per-band weights as edge features.

Graphs are stored densely and turned into PyG edge lists one batch at a time on
the device, so 40k windows train quickly without building 40k Data objects.

Matrix convention (as in connectivity.py): W[source, target]; PyG messages flow
source -> target, so PDC keeps its direction. wPLI is symmetric (both directions).
"""
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from . import graph_metrics as gm


# ------------------------------------------------------------------ graphs
def filtered_graphs(conn, directed, lo, hi):
    """(n_win, n_bands, C, C) -> same shape, filtered per window and band, scaled to [0, 1]
    by the window's strongest edge (as in graph_metrics.graph_features)."""
    out = np.zeros_like(conn)
    for w in range(conn.shape[0]):
        for b in range(conn.shape[1]):
            W = gm.filter_graph(conn[w, b], lo, hi, directed)
            mx = W.max()
            out[w, b] = W / mx if mx > 0 else W
    return out


def node_features(bandpower, graphs, kind):
    """bandpower (n_win, n_bands, C) -> (n_win, C, F).
    kind 'bp': log band power per band. 'bp+conn': plus each node's outgoing edge
    weights in every band (its connectivity profile)."""
    x = np.transpose(bandpower, (0, 2, 1))                      # (n_win, C, n_bands)
    if kind == "bp+conn":
        prof = np.transpose(graphs, (0, 2, 1, 3)).reshape(len(graphs), graphs.shape[2], -1)
        x = np.concatenate([x, prof], axis=2)
    return x.astype(np.float32)


def to_batch(x, A):
    """Dense batch -> PyG tensors. x: (B, C, F), A: (B, n_bands, C, C) on the device.
    An edge i -> j exists if it was kept in any band; its features are the band weights."""
    B, C, _ = x.shape
    b, i, j = torch.nonzero(A.amax(dim=1) > 0, as_tuple=True)
    edge_index = torch.stack([b * C + i, b * C + j])
    edge_attr = A[b, :, i, j]                                    # (E, n_bands)
    batch = torch.arange(B, device=x.device).repeat_interleave(C)
    return x.reshape(B * C, -1), edge_index, edge_attr, batch


# ------------------------------------------------------------------ model
class GAT(nn.Module):
    """GATv2 layers (edge-aware attention) -> mean + max pooling -> MLP classifier."""

    def __init__(self, in_dim, edge_dim, n_classes, hidden=32, heads=4, layers=2, dropout=0.3):
        super().__init__()
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.convs.append(GATv2Conv(d, hidden, heads=heads, edge_dim=edge_dim,
                                        dropout=dropout, add_self_loops=True, fill_value="mean"))
            self.norms.append(nn.LayerNorm(hidden * heads))
            d = hidden * heads
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(2 * d, hidden), nn.ELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_classes))

    def forward(self, x, edge_index, edge_attr, batch, return_attention=False):
        att = None
        for k, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = nn.functional.dropout(x, self.dropout, self.training)
            if return_attention and k == len(self.convs) - 1:
                x, (ei, alpha) = conv(x, edge_index, edge_attr, return_attention_weights=True)
                att = (ei, alpha.mean(dim=1))                    # average over heads
            else:
                x = conv(x, edge_index, edge_attr)
            x = nn.functional.elu(norm(x))
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        logits = self.head(g)
        return (logits, att) if return_attention else logits
