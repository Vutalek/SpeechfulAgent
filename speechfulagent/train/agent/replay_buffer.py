from collections import deque
from typing import List

import numpy as np

from speechfulagent.dataclasses import Experience


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        idxs = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[idx] for idx in idxs]