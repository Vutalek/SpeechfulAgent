import json
from dataclasses import dataclass, asdict
import time
import logging
logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s]: %(message)s")
logger = logging.getLogger("train")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from numpy.typing import NDArray
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()


@dataclass
class Experience:
    state: int | NDArray
    action: int | NDArray
    reward: float
    next_state: int | NDArray
    done: bool

    dict = asdict

def json_to_episode(file: str) -> list[Experience]:
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

def proccess_folder(path: str):
    episodes = []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for expl in data:
        ep_folder = expl["folder"]
        ep_file = expl["filename"]
        episode = json_to_episode(ep_folder + "/" + ep_file)
        episodes.append(
            {
                "episode": episode,
                "explanation": expl["explanation"]
            }
        )
    return episodes

class ExperienceDataset(Dataset):
    def __init__(self, folder: str, tokenizer, device: str="cpu"):
        self.episodes = []
        self.episodes.extend(proccess_folder(folder + "/explanations_bad.json"))
        self.episodes.extend(proccess_folder(folder + "/explanations_good.json"))
        self.tokenizer = tokenizer
        self.device = device

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
    
class StateEncoder(nn.Module):
    def __init__(
        self,
        module_size: int=32,
        hidden_size: int=512,
        projected_size: int=2048,
    ):
        super().__init__()
        self.module_size = module_size
        self.hidden_size = hidden_size
        self.projected_size = projected_size

        self.state_mod = nn.Linear(16, module_size)
        self.action_mod = nn.Linear(4, module_size)
        self.reward_mod = nn.Linear(1, module_size)

        self.lstm = nn.LSTM(input_size=3*module_size, hidden_size=hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.project = nn.Linear(hidden_size, projected_size)

    def forward(self, states, actions, rewards):
        states = self.state_mod(F.one_hot(states, 16).to(dtype=torch.float32))
        actions = self.action_mod(F.one_hot(actions, 4).to(dtype=torch.float32))
        rewards = self.reward_mod(rewards.view((-1, 1)))
        exp_tensor = torch.concat([states, actions, rewards], dim=1)
        
        lstm_out, (h_n, c_n) = self.lstm(exp_tensor)
        normed = self.norm(lstm_out)
        projected = self.project(normed)
        return projected
    
class Explainer():
    def __init__(self, transformer):
        self.transformer = transformer

    def _apply_top_k(self, logits: torch.Tensor, k: int=0) -> torch.Tensor:
        if k == 0:
            return logits
        top_logits, top_ids = torch.topk(logits, k, dim=-1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[top_ids] = False
        return torch.masked_fill(logits, mask=mask, value=float("-inf"))
    
    def _apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        return logits / temperature
    
    def _greedy_sampling(self, logits: torch.Tensor) -> int:
        val, ind = torch.max(logits, dim=0)
        return int(ind.item())
    
    def _random_sampling(self, logits: torch.Tensor) -> int:
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())
    
    def generate_with_loss(
        self,
        input_embeds: torch.Tensor,
        startage: torch.Tensor,
        endage: torch.Tensor,
        startas: torch.Tensor,
        ground_truth
    ) -> torch.Tensor:
        if self.transformer is None:
            raise RuntimeError("Model not loaded!")
        
        vector_ground_truth = self.transformer.model.embed_tokens(
            ground_truth.to(self.transformer.device)
        )
        
        llm_input = torch.cat([
            startage.squeeze(0), 
            input_embeds.squeeze(0),
            endage.squeeze(0),
            startas.squeeze(0),
            vector_ground_truth.squeeze(0) # Teacher Forcing
        ], dim=0).unsqueeze(0)

        ignore_index = -100
        labels = torch.full((llm_input.shape[0], llm_input.shape[1]), ignore_index, dtype=torch.long, device=ground_truth.device)
        prefix_len = startage.shape[1] + input_embeds.shape[1] + endage.shape[1] + startas.shape[1]
        labels[0, prefix_len:] = ground_truth

        outputs = self.transformer.forward(
            inputs_embeds=llm_input.to(torch.bfloat16), 
            labels=labels,
            use_cache=False
        )

        return outputs.loss


if __name__ == "__main__":
    device_qwen = "cuda"
    device_se = "cuda"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B").to(device_qwen)
    qwen.gradient_checkpointing_enable() 
    qwen.config.use_cache = False
    explainer = Explainer(qwen)

    with torch.no_grad():
        startage = explainer.transformer.model.embed_tokens(torch.LongTensor([[151644, 872, 198]]).to(device_qwen))
        endage = explainer.transformer.model.embed_tokens(torch.LongTensor([[151645, 198]]).to(device_qwen))
        startas = explainer.transformer.model.embed_tokens(torch.LongTensor([[151644, 77091, 198]]).to(device_qwen))

    dataset = ExperienceDataset("final_dataset", tokenizer, device=device_se)
    train_set, validation_set = random_split(dataset, [450, 50])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False)

    se = StateEncoder().to(device_se)
    se.train()
    se_optim = optim.AdamW(se.parameters(), lr=1e-3, weight_decay=5e-4)
    se_scheduler = optim.lr_scheduler.CosineAnnealingLR(se_optim, T_max=1000)

    loss_history = []
    validation_history = []
    best_validation_loss = None
    early_stopping_counter = 0
    early_stopping_patience = 11
    logger.info("start training")
    for epoch in range(1000):
        # training
        se.train()
        history = []
        for states, actions, rewards, explanation in train_loader:
            start = time.time()

            states = states.squeeze(0)
            actions = actions.squeeze(0)
            rewards = rewards.squeeze(0)
            explanation = explanation.squeeze(0).to(device_qwen)

            se_optim.zero_grad()
            state_embeds = se.forward(states, actions, rewards).to(device_qwen)
            loss = explainer.generate_with_loss(
                state_embeds.unsqueeze(0).to(dtype=torch.bfloat16), 
                startage, endage, startas,
                explanation
            )

            loss.backward()
            total_norm = clip_grad_norm_(se.parameters(), max_norm=float('inf'), norm_type=2)
            se_optim.step()

            logger.info(f"TRAIN epoch: {epoch} loss: {loss.item():.4f}, grad_norm: {total_norm.item()}, lr: {se_scheduler.get_last_lr()[0]:.6f}, time: {time.time() - start:.2f}s")
            history.append(loss.item())
        logger.info(f"TRAIN epoch summary: {epoch} loss: {sum(history) / len(history):.4f}")
        se_scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        loss_history.append(history)

        # validation
        se.eval()
        history = []
        with torch.no_grad():
            for states, actions, rewards, explanation in validation_loader:
                states = states.squeeze(0)
                actions = actions.squeeze(0)
                rewards = rewards.squeeze(0)
                explanation = explanation.squeeze(0)

                state_embeds = se.forward(states, actions, rewards).to(device_qwen)
                loss = explainer.generate_with_loss(
                    state_embeds.unsqueeze(0).to(dtype=torch.bfloat16), 
                    startage, endage, startas,
                    explanation
                )

                history.append(loss.item())
            logger.info(f"VALIDATION epoch: {epoch} loss: {sum(history) / len(history):.4f}")
            validation_history.append(history)

            #early_stopping
            current_loss = sum(history) / len(history)
            if best_validation_loss is None or current_loss < best_validation_loss:
                best_validation_loss = current_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                logger.warning("early stopping triggered")
                break
    with open("weights.pth", "wb") as f:
        torch.save(se.state_dict(), f)
    with open("loss_history.json", "wt") as f:
        json.dump(loss_history, f)
    with open("validation_history.json", "wt") as f:
        json.dump(validation_history, f)
    logger.info("training finished")