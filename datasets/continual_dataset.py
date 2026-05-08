"""
datasets/continual_dataset.py
Returns (user, pos_item, neg_item, context_vec) where
context_vec = [time_idx, activity_idx, age_idx, weekday, mood_idx]
"""
import torch
from torch.utils.data import Dataset
import numpy as np

class ContinualDataset(Dataset):
    def __init__(self, interactions: np.ndarray, n_items: int):
        self.data    = interactions
        self.n_items = n_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row  = self.data[idx]
        user = int(row[0]); item = int(row[1])
        ctx  = row[2:].astype(np.int64)
        neg  = np.random.randint(0, self.n_items)
        while neg == item:
            neg = np.random.randint(0, self.n_items)
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(neg,  dtype=torch.long),
            torch.tensor(ctx,  dtype=torch.long),
        )
