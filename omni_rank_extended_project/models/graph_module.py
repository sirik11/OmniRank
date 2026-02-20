"""
graph_module.py
----------------

This module demonstrates how to build a social graph from user interactions and train a simple
graph neural network (GNN) to produce user embeddings.  These embeddings capture network
structure and can be combined with behavioural embeddings from the two‑tower recommender.

We use **NetworkX** to construct the graph and **DGL** for GNN training.  If you prefer
PyTorch Geometric (PyG), you can swap out the DGL code.
"""

import argparse
import os
import pandas as pd
import numpy as np
import networkx as nx

try:
    import dgl
    import dgl.nn as dglnn
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    dgl = None


def build_user_graph(events_path: str, users_path: str, threshold: int = 10) -> nx.Graph:
    """Construct an undirected graph where edges connect users who interact with similar posts.

    Args:
        events_path: Path to Parquet file containing interactions.
        users_path: Path to Parquet file containing user metadata.
        threshold: Minimum number of common posts for an edge between two users.
    Returns:
        A NetworkX graph object.
    """
    events = pd.read_parquet(events_path)
    # Create a mapping from post to users who interacted
    post_to_users = events.groupby('post_id')['user_id'].apply(set)
    # Compute user co‑interaction counts
    edge_counts = {}
    for users in post_to_users:
        users_list = list(users)
        for i in range(len(users_list)):
            for j in range(i + 1, len(users_list)):
                pair = (users_list[i], users_list[j]) if users_list[i] < users_list[j] else (users_list[j], users_list[i])
                edge_counts[pair] = edge_counts.get(pair, 0) + 1
    G = nx.Graph()
    # Add nodes with user features
    users_df = pd.read_parquet(users_path)
    for _, row in users_df.iterrows():
        G.add_node(row['user_id'], account_age=row['account_age_days'], num_friends=row['num_friends'])
    # Add edges if co‑interaction count exceeds threshold
    for (u, v), count in edge_counts.items():
        if count >= threshold:
            G.add_edge(u, v, weight=count)
    return G


class GCN(nn.Module):
    """A simple 2‑layer Graph Convolutional Network."""
    def __init__(self, in_feats: int, hidden_size: int, out_feats: int):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_size)
        self.conv2 = dglnn.GraphConv(hidden_size, out_feats)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        return x


def train_gnn(g: nx.Graph, epochs: int = 50, hidden_size: int = 32, out_feats: int = 16) -> np.ndarray:
    """Train a small GCN on user graphs and return the learned embeddings."""
    if dgl is None:
        raise ImportError("DGL is not installed.  Install dgl to use GNN functionality.")
    g_dgl = dgl.from_networkx(g, node_attrs=['account_age', 'num_friends'])
    # Feature matrix: normalise and concatenate features
    feats = torch.tensor(
        np.stack([
            g_dgl.ndata['account_age'].float() / 365.0,
            g_dgl.ndata['num_friends'].float() / 1000.0,
        ], axis=1),
        dtype=torch.float32
    )
    model = GCN(in_feats=feats.shape[1], hidden_size=hidden_size, out_feats=out_feats)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        model.train()
        optimiser.zero_grad()
        embeddings = model(g_dgl, feats)
        # Self‑supervised objective: preserve neighbourhood similarity (contrastive)
        pos_loss = 0.0
        num_edges = g_dgl.num_edges()
        # Negative sampling: random node pairs
        # We avoid explicit loops for efficiency on real datasets by using adjacency matrices and negative sampling from them
        # Here we keep it simple for demonstration
        for u, v in zip(*g_dgl.edges()):
            u_emb = embeddings[u]
            v_emb = embeddings[v]
            pos_loss += -torch.log(torch.sigmoid(torch.dot(u_emb, v_emb)))
        # Sample random pairs
        rand_u = torch.randint(0, g_dgl.num_nodes(), (num_edges,))
        rand_v = torch.randint(0, g_dgl.num_nodes(), (num_edges,))
        neg_loss = -torch.log(1 - torch.sigmoid((embeddings[rand_u] * embeddings[rand_v]).sum(dim=1))).sum()
        loss = (pos_loss + neg_loss) / num_edges
        loss.backward()
        optimiser.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: loss {loss.item():.4f}")
    return embeddings.detach().numpy()


def main():
    parser = argparse.ArgumentParser(description="Train a graph neural network on user social graph.")
    parser.add_argument('--events_path', type=str, required=True)
    parser.add_argument('--users_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default='models/gnn', help='Directory to save embeddings')
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    print("Building user graph...")
    G = build_user_graph(args.events_path, args.users_path)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    embeddings = train_gnn(G, epochs=args.epochs)
    np.save(os.path.join(args.model_dir, 'user_gnn_embeddings.npy'), embeddings)
    print(f"Saved embeddings to {args.model_dir}/user_gnn_embeddings.npy")


if __name__ == '__main__':
    main()