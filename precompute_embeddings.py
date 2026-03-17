"""
precompute_embeddings.py
-------------------------
Ginger Agents — CS5180 Spring 2026

Pre-compute GNN topology embeddings for all Rome graphs.
Run once → save to rome_embeddings.pt → PPO uses fixed embeddings + dynamic coords.

Usage:
    python precompute_embeddings.py --data_path rome_gnn_data.pt --hidden_dim 64
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ── GNN Encoder (topology only) ──────────────────────────────────────────────

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class TopologyGCN(nn.Module):
    """
    GCN that encodes ONLY topology features (degree, clustering, neighbor structure).
    Coordinates are excluded — they change during RL, topology doesn't.

    Input:  [N, 3] = (normalized_degree, clustering_coeff, avg_neighbor_dist)
    Output: [N, hidden_dim] node embeddings, [hidden_dim] graph embedding
    """

    def __init__(self, in_dim=3, hidden_dim=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            if HAS_PYG:
                self.convs.append(GCNConv(in_ch, hidden_dim))
            else:
                self.convs.append(ManualGCNLayer(in_ch, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.has_pyg = HAS_PYG

    def forward(self, x, edge_index):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            if self.has_pyg:
                h = conv(h, edge_index)
            else:
                adj = edge_index_to_adj(edge_index, x.size(0))
                h = conv(h, adj)
            h = norm(h)
            h = F.relu(h)
        node_emb = self.out_proj(h)        # [N, hidden_dim]
        graph_emb = node_emb.mean(dim=0)   # [hidden_dim]
        return node_emb, graph_emb


class ManualGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj):
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        adj_norm = adj * deg_inv_sqrt * deg_inv_sqrt.T
        return adj_norm @ self.W(x) + self.bias


def edge_index_to_adj(edge_index, num_nodes):
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj += torch.eye(num_nodes)
    return adj


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-compute GNN topology embeddings")
    parser.add_argument("--data_path", type=str, default="rome_gnn_data.pt")
    parser.add_argument("--output_path", type=str, default="rome_embeddings.pt")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load data
    print(f"Loading {args.data_path}...")
    data_list = torch.load(args.data_path, weights_only=False)
    print(f"  {len(data_list)} graphs loaded")

    # Build encoder
    encoder = TopologyGCN(in_dim=3, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    encoder.eval()
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"  Using {'PyG' if HAS_PYG else 'Manual'} GCN\n")

    # Pre-compute embeddings
    results = []
    start = time.time()

    with torch.no_grad():
        for d in tqdm(data_list, desc="Embedding graphs", unit="graph"):
            # Topology-only features: columns 2,3,4 from x = (degree, clustering, avg_dist)
            topo_features = d["x"][:, 2:]  # [N, 3]
            edge_index = d["edge_index"]

            node_emb, graph_emb = encoder(topo_features, edge_index)

            results.append({
                "graph_name":    d["graph_name"],
                "node_emb":      node_emb,          # [N, hidden_dim] fixed topology embedding
                "graph_emb":     graph_emb,          # [hidden_dim]
                "edge_index":    d["edge_index"],    # keep for reference
                "coords":        d["coords"],        # initial coords
                "num_nodes":     d["num_nodes"],
                "num_edges":     d["num_edges"],
                "y_crossings":   d["y_crossings"],
            })

    elapsed = time.time() - start

    # Save
    torch.save(results, args.output_path)

    print(f"\n{'=' * 60}")
    print(f"DONE!  {args.output_path}")
    print(f"{'=' * 60}")
    print(f"  Graphs embedded    : {len(results)}")
    print(f"  Time               : {elapsed:.1f}s")
    print(f"  Node emb shape     : {results[0]['node_emb'].shape}")
    print(f"  Graph emb shape    : {results[0]['graph_emb'].shape}")
    print(f"  File size          : {os.path.getsize(args.output_path) / 1e6:.1f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
