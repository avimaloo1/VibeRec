"""
continual/replay_buffer.py
Reservoir-sampling replay buffer with mood x activity stratified sampling.
Row format: (user, item, time_idx, activity_idx, age_idx, weekday, mood_idx) — 7 cols.
"""
import numpy as np
import random
from collections import defaultdict

class ReplayBuffer:
    def __init__(self, size=8_000):
        self.size   = size
        self.buffer = []
        self._count = 0

    def add(self, task_data):
        for row in task_data:
            self._count += 1
            if len(self.buffer) < self.size:
                self.buffer.append(row.copy())
            else:
                j = random.randint(0, self._count-1)
                if j < self.size:
                    self.buffer[j] = row.copy()

    def sample(self, n):
        n = min(n, len(self.buffer))
        if n == 0: return np.empty((0,7), dtype=np.int64)
        return np.stack(random.sample(self.buffer, n))

    def sample_stratified(self, n):
        """Balance replay across activity x mood buckets."""
        if not self.buffer: return np.empty((0,7), dtype=np.int64)
        buckets = defaultdict(list)
        for row in self.buffer:
            buckets[(int(row[3]), int(row[6]))].append(row)
        per_bucket = max(1, n // len(buckets))
        rows = []
        for brows in buckets.values():
            k = min(per_bucket, len(brows))
            rows.extend(random.sample(brows, k))
        if len(rows) < n and len(self.buffer) > len(rows):
            extra = random.sample(self.buffer, min(n-len(rows), len(self.buffer)))
            rows.extend(extra)
        random.shuffle(rows)
        return np.stack(rows[:n])

    def __len__(self):
        return len(self.buffer)
