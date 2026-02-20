"""
policy_controller.py
---------------------

This module implements a simple contextual bandit controller for ranking policy selection.
Each policy arm (e.g., a ranking strategy) has a linear model that predicts expected reward
given a context vector.  We use the **LinUCB** algorithm with confidence bounds to
balance exploration and exploitation.  It can be extended to Thompson Sampling or full
reinforcement learning with off‑policy evaluation.
"""

import numpy as np


class LinUCBPolicy:
    """LinUCB algorithm for a single policy arm."""

    def __init__(self, context_dim: int, alpha: float = 1.0):
        self.context_dim = context_dim
        self.alpha = alpha
        self.A = np.eye(context_dim)  # Feature covariance
        self.b = np.zeros(context_dim)  # Cumulative reward-weighted context

    def update(self, context: np.ndarray, reward: float):
        self.A += np.outer(context, context)
        self.b += reward * context

    def predict(self, context: np.ndarray) -> float:
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        p = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
        return p


class PolicyController:
    """Manages multiple policies and selects one according to contextual bandits."""

    def __init__(self, num_policies: int, context_dim: int, alpha: float = 1.0):
        self.policies = [LinUCBPolicy(context_dim, alpha) for _ in range(num_policies)]
        self.num_policies = num_policies
        self.rewards = [0.0] * num_policies
        self.counts = [0] * num_policies

    def select_policy(self, context: np.ndarray) -> int:
        scores = [policy.predict(context) for policy in self.policies]
        best_policy = int(np.argmax(scores))
        return best_policy

    def update_policy(self, policy_idx: int, context: np.ndarray, reward: float):
        self.policies[policy_idx].update(context, reward)
        self.rewards[policy_idx] += reward
        self.counts[policy_idx] += 1

    def report(self):
        for i in range(self.num_policies):
            mean_reward = self.rewards[i] / self.counts[i] if self.counts[i] else 0.0
            print(f"Policy {i}: pulled {self.counts[i]} times, mean reward {mean_reward:.4f}")