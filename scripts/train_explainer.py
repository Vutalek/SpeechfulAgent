from typing import Dict, List, Any
import logging
logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s]: %(message)s")
logger = logging.getLogger("test")

import torch
import torch.nn.functional as F

from speechfulagent.dataclasses import Experience
from speechfulagent.train.explainer import R3LExplainerTrainer


def collate_tensors(
    prompt: List[Dict[str, torch.Tensor]],
    context: List[Experience] | torch.Tensor | Any
) -> List[torch.Tensor]:
    states = torch.Tensor([])
    actions = torch.Tensor([])
    rewards = torch.Tensor([])
    for exp in prompt:
        state = torch.as_tensor([exp["state"]], dtype=torch.long)
        state = F.one_hot(state, 16)
        states = torch.concat([states, state], dim=0)

        action = torch.as_tensor([exp["action"]], dtype=torch.long)
        action = F.one_hot(action, 4)
        actions = torch.concat([actions, action], dim=0)

        reward = torch.as_tensor([exp["reward"]], dtype=torch.float32)
        rewards = torch.concat([rewards, reward], dim=0)
    return [states, actions, rewards.view((-1, 1))]

trainer = R3LExplainerTrainer(
    pathfile="final_dataset",
    collation_fn=collate_tensors,
    max_iter=1,
    logger=logger
)
explainer, train_info, losses = trainer.train()
explainer.save_model("explainer", train=train_info)
print(losses)