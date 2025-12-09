import json
import random
from typing import Any

import torch
from torch.utils.data import Dataset

from speechfulagent.dataclasses import Experience
from speechfulagent.explainer.preprocessing import Tokenizer, embed_sequence


class SequenceExplanationsDataset(Dataset):
    def __init__(self, pathfile: str, max_length: int, seed: int):
        random.seed(seed)
        with open(pathfile, "rt") as f:
            self.raw_data = json.load(f)
        for sample in self.raw_data:
            experiences = [Experience(**exp) for exp in sample["sequence"]]
            sample["sequence"] = experiences

        self.tokenizer = Tokenizer()
        corpus = []
        for sample in self.raw_data:
            corpus.extend(sample["explanation"])
        self.tokenizer.build_vocab(corpus)
        self.max_length = max_length

        self.sequences = []
        self.tails = []
        self.explanations = []
        for sample in self.raw_data:
            seq, tail = embed_sequence(sample["sequence"], sample["tail"]+1)
            self.sequences.append(seq)
            self.tails.append(tail)
            self.explanations.append(
                [self.tokenizer.encode(explain, max_length) for explain in sample["explanation"]]
            )
        
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index) -> Any:
        explanation = random.choice(range(len(self.explanations[index])))
        return self.sequences[index], self.tails[index], torch.Tensor(self.explanations[index][explanation])
    