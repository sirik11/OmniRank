"""
generate_large_scale_data.py
---------------------------------

This script generates synthetic user, post, and interaction data for the extended feed ranking platform.  It supports creating
datasets of arbitrary size by sampling from configurable distributions.  The output is stored in Parquet files to ensure fast
loading and compatibility with distributed processing frameworks.

Usage:

```bash
python data_generation/generate_large_scale_data.py \
    --users 1000000 \
    --posts 5000000 \
    --days 30 \
    --events 50000000 \
    --output_dir data
```

You can also specify `--seed` for reproducibility.
"""

import argparse
import os
import random
import numpy as np
import pandas as pd

from datetime import datetime, timedelta


def generate_users(num_users: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_ids = np.arange(1, num_users + 1)
    # Example user features: account age in days, number of friends
    account_age = rng.integers(0, 365 * 5, size=num_users)
    num_friends = rng.poisson(100, size=num_users)
    languages = rng.choice(['en', 'es', 'fr', 'de', 'zh', 'hi'], size=num_users)
    return pd.DataFrame({
        'user_id': user_ids,
        'account_age_days': account_age,
        'num_friends': num_friends,
        'language': languages,
    })


def generate_posts(num_posts: int, seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    post_ids = np.arange(1, num_posts + 1)
    # Assign categories and quality scores
    categories = rng.choice(['tech', 'sports', 'fashion', 'music', 'travel', 'food', 'finance'], size=num_posts)
    quality = rng.uniform(0, 1, size=num_posts)
    return pd.DataFrame({
        'post_id': post_ids,
        'category': categories,
        'quality_score': quality,
    })


def generate_events(users: pd.DataFrame, posts: pd.DataFrame, num_events: int,
                    start_date: datetime, end_date: datetime, seed: int = 44) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_ids = users['user_id'].to_numpy()
    post_ids = posts['post_id'].to_numpy()
    categories = posts['category'].to_numpy()
    # Sample user and post pairs
    sampled_users = rng.choice(user_ids, size=num_events)
    sampled_posts = rng.choice(post_ids, size=num_events)
    # Generate timestamps uniformly over the interval
    total_seconds = (end_date - start_date).total_seconds()
    random_seconds = rng.uniform(0, total_seconds, size=num_events)
    timestamps = [start_date + timedelta(seconds=float(s)) for s in random_seconds]
    # Randomly assign event types (e.g., view, like, comment)
    event_types = rng.choice(['view', 'like', 'comment', 'share'], size=num_events, p=[0.7, 0.2, 0.08, 0.02])
    # Engagement score could be derived from post quality and event type
    post_quality_map = dict(zip(posts['post_id'], posts['quality_score']))
    engagement_scores = [post_quality_map[pid] * (1 + 0.5 * (et != 'view')) for pid, et in zip(sampled_posts, event_types)]
    return pd.DataFrame({
        'user_id': sampled_users,
        'post_id': sampled_posts,
        'timestamp': timestamps,
        'event_type': event_types,
        'engagement_score': engagement_scores,
    })


def main():
    parser = argparse.ArgumentParser(description="Generate large‑scale synthetic data for the feed ranking platform.")
    parser.add_argument('--users', type=int, default=100000, help='Number of users to generate')
    parser.add_argument('--posts', type=int, default=500000, help='Number of posts to generate')
    parser.add_argument('--days', type=int, default=14, help='Number of days of activity')
    parser.add_argument('--events', type=int, default=2000000, help='Number of interaction events')
    parser.add_argument('--output_dir', type=str, default='data', help='Directory to store Parquet files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating {args.users} users...")
    users_df = generate_users(args.users, seed=args.seed)
    users_path = os.path.join(args.output_dir, 'users.parquet')
    users_df.to_parquet(users_path)
    print(f"Saved users to {users_path}")

    print(f"Generating {args.posts} posts...")
    posts_df = generate_posts(args.posts, seed=args.seed + 1)
    posts_path = os.path.join(args.output_dir, 'posts.parquet')
    posts_df.to_parquet(posts_path)
    print(f"Saved posts to {posts_path}")

    print(f"Generating {args.events} events...")
    start_date = datetime.now() - timedelta(days=args.days)
    end_date = datetime.now()
    events_df = generate_events(users_df, posts_df, args.events, start_date, end_date, seed=args.seed + 2)
    events_path = os.path.join(args.output_dir, 'events.parquet')
    events_df.to_parquet(events_path)
    print(f"Saved events to {events_path}")


if __name__ == '__main__':
    main()