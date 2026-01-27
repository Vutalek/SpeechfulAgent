from typing import Tuple, List

import torch
import torch.nn.functional as F

from speechfulagent.dataclasses import Experience


def embed_experience(exp: Experience) -> torch.Tensor:
    """FOR FROZENLAKE ONLY"""
    embedding = [0.0] * 37
    if exp.state != -1:
        embedding[exp.state] = 1.0
        embedding[16 + exp.action] = 1.0
        embedding[20 + exp.next_state] = 1.0
        embedding[-1] = exp.reward
    return torch.Tensor(embedding).view(1, -1)

def embed_sequence(sequence: List[Experience], tail: int, length: int=10) -> Tuple[torch.Tensor, torch.Tensor]:
    """FOR FROZENLAKE ONLY"""
    embeds = [embed_experience(exp) for exp in sequence]
    for i in range(len(sequence)):
        if sequence[i].state == -1:
            length = i
            break
    full_seq_embed = torch.concat(embeds, dim=0)
    tail_embed = torch.concat(embeds[(length-tail):length], dim=0)
    tail_embed = F.pad(tail_embed, (0, 0, 0, length-tail))
    return full_seq_embed, tail_embed