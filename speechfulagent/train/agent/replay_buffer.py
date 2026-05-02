"""Module with replay buffer for training."""

from collections import deque
from typing import List

import numpy as np

from speechfulagent.dataclasses import Experience


class ReplayBuffer:
    """Implementation of replay buffer."""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        """Add experience in buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch_size experiences from buffer."""
        idxs = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[idx] for idx in idxs]
