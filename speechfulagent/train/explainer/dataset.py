import json
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from speechfulagent.dataclasses import Experience


class ExperienceDataset(Dataset):
    def __init__(self, folder: str, tokenizer, device: str="cpu"):
        self.episodes = []
        self.episodes.extend(self._proccess_folder(folder + "/explanations_bad.json"))
        self.episodes.extend(self._proccess_folder(folder + "/explanations_good.json"))
        self.tokenizer = tokenizer
        self.device = device

    def _json_to_episode(self, file: str) -> List[Experience]:
        with open(file, "rt") as f:
            episode = json.load(f)
        experiences = []
        for step in episode:
            experiences.append(
                Experience(
                    state=step["state"],
                    action=step["action"],
                    reward=step["reward"],
                    next_state=None,
                    done=False
                )
            )
        experiences[-2].done = True
        for i in range(len(experiences)-1):
            experiences[i].next_state = experiences[i+1].state
        return experiences[:-1]
    
    def _proccess_folder(self, path: str) -> List[Dict[str, Any]]:
        episodes = []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for expl in data:
            ep_folder = expl["folder"]
            ep_file = expl["filename"]
            episode = self._json_to_episode(ep_folder + "/" + ep_file)
            episodes.append(
                {
                    "episode": episode,
                    "explanation": expl["explanation"]
                }
            )
        return episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        entry = self.episodes[idx]
        episode = entry["episode"]
        text = entry["explanation"]
        states = torch.as_tensor([exp.state for exp in episode], dtype=torch.long).to(self.device)
        actions = torch.as_tensor([exp.action for exp in episode], dtype=torch.long).to(self.device)
        rewards = torch.as_tensor([exp.reward for exp in episode], dtype=torch.float32).to(self.device)
        tok_exp = self.tokenizer.encode(text)
        explanation = torch.as_tensor(tok_exp, dtype=torch.long).to(self.device),
        return states, actions, rewards, explanation[0]