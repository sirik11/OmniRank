"""
two_tower_recommender.py
------------------------

This module implements a simple two‑tower recommender architecture using PyTorch.  The model
embeds users and items into a shared latent space and optimises an implicit feedback loss.
The implementation is intentionally modular: you can swap the encoders for more sophisticated
architectures (e.g., transformers, CNNs for images) and integrate content features as needed.

Example usage:

```bash
python -m models.two_tower_recommender \
    --train_path data/events.parquet \
    --users_path data/users.parquet \
    --posts_path data/posts.parquet \
    --model_dir models/checkpoints \
    --epochs 5
```

This script only trains on a small sample by default; for large datasets you should integrate
with PyTorch DataLoader and potentially distributed training frameworks.
"""

import argparse
import os
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class InteractionDataset(Dataset):
    """A simple dataset that samples positive interactions and negative samples."""

    def __init__(self, events_df: pd.DataFrame, num_items: int, negative_ratio: int = 3):
        self.user_ids = events_df['user_id'].to_numpy()
        self.item_ids = events_df['post_id'].to_numpy()
        self.num_items = num_items
        self.negative_ratio = negative_ratio

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        user = self.user_ids[idx]
        pos_item = self.item_ids[idx]
        # Negative sampling: sample a random item that's different from the positive item
        neg_item = pos_item
        while neg_item == pos_item:
            neg_item = torch.randint(1, self.num_items + 1, (1,)).item()
        return user, pos_item, neg_item


class TwoTowerModel(nn.Module):
    """Two‑tower recommender with separate user and item encoders."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim)
        self.user_bias = nn.Embedding(num_users + 1, 1)
        self.item_bias = nn.Embedding(num_items + 1, 1)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u_e = self.user_embedding(users)
        i_e = self.item_embedding(items)
        dot = (u_e * i_e).sum(dim=1) + self.user_bias(users).squeeze() + self.item_bias(items).squeeze()
        return dot


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian personalised ranking loss."""
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()


def train_loop(model: nn.Module, dataloader: DataLoader, optimiser: torch.optim.Optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    for users, pos_items, neg_items in dataloader:
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        optimiser.zero_grad()
        pos_scores = model(users, pos_items)
        neg_scores = model(users, neg_items)
        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(users)
    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train a two‑tower recommender model.")
    parser.add_argument('--train_path', type=str, required=True, help='Parquet file containing interaction events')
    parser.add_argument('--users_path', type=str, required=True, help='Parquet file containing user metadata')
    parser.add_argument('--posts_path', type=str, required=True, help='Parquet file containing post metadata')
    parser.add_argument('--model_dir', type=str, default='models/checkpoints', help='Directory to save model weights')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Latent dimension for embeddings')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on')
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    print("Loading data...")
    events = pd.read_parquet(args.train_path)
    users_df = pd.read_parquet(args.users_path)
    posts_df = pd.read_parquet(args.posts_path)
    num_users = users_df['user_id'].max()
    num_items = posts_df['post_id'].max()
    dataset = InteractionDataset(events, num_items)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = TwoTowerModel(num_users, num_items, embedding_dim=args.embedding_dim).to(args.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train_loop(model, dataloader, optimiser, torch.device(args.device))
        print(f"Epoch {epoch + 1}: Loss {loss:.4f}")
        # Save checkpoint
        checkpoint_path = os.path.join(args.model_dir, f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    # Export embeddings for offline serving (optional)
    user_embeddings = model.user_embedding.weight.detach().cpu().numpy()
    item_embeddings = model.item_embedding.weight.detach().cpu().numpy()
    np.save(os.path.join(args.model_dir, 'user_embeddings.npy'), user_embeddings)
    np.save(os.path.join(args.model_dir, 'item_embeddings.npy'), item_embeddings)


if __name__ == '__main__':
    main()