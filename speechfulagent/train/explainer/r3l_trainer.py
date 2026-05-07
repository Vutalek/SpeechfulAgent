import time
from typing import List, Tuple, Dict, Any

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from speechfulagent.dataclasses import R3LTrainInfo
from speechfulagent.train.explainer import BaseExplainerTrainer, ExperienceDataset
from speechfulagent.explainer.r3l import R3LExplainer, R3LStateEncoder


class R3LExplainerTrainer(BaseExplainerTrainer):
    def __init__(
        self,
        pathfile: str,
        validation_fraction: float=0.1,
        modules: List[int]=[16, 4, 1],
        module_size: int=32,
        hidden_size: int=512,
        projected_size: int=2048,
        learning_rate: float=1e-3,
        weight_decay: float=5e-4,
        llm_model_name: str="Qwen/Qwen3-1.7B",
        llm_tokenizer_name: str="Qwen/Qwen3-1.7B",
        llm_think_start: List[int]=[151644, 872, 198],
        llm_think_end: List[int]=[151645, 198],
        llm_text_start: List[int]=[151644, 77091, 198],
        encoder_device: str="cpu",
        llm_device: str="cpu",
        max_iter: int=100,
        early_stopping: int=5,
        logger=None,
        seed: int=70
    ):
        super().__init__(seed)
        self.pathfile = pathfile
        self.validation_fraction = validation_fraction
        self.modules = modules
        self.module_size = module_size
        self.hidden_size = hidden_size
        self.projected_size = projected_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.llm_model_name = llm_model_name
        self.llm_tokenizer_name = llm_tokenizer_name
        self.llm_think_start = llm_think_start
        self.llm_think_end = llm_think_end
        self.llm_text_start = llm_text_start
        self.encoder_device = encoder_device
        self.llm_device = llm_device
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.logger = logger
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_tokenizer_name)
        self.llm = AutoModelForCausalLM.from_pretrained(self.llm_model_name).to(self.llm_device)
        self.llm.gradient_checkpointing_enable() 
        self.llm.config.use_cache = False
        
        self.dataset = ExperienceDataset(self.pathfile, self.tokenizer, device=self.encoder_device)
        train_size = int(len(self.dataset) * (1 - self.validation_fraction))
        val_size = len(self.dataset) - train_size
        self.train_set, self.validation_set = random_split(self.dataset, [train_size, val_size])

        self.encoder = R3LStateEncoder(
            self.modules,
            self.module_size,
            self.hidden_size,
            self.projected_size
        ).to(self.encoder_device)
        self.encoder.train()

        self.explainer = R3LExplainer()
        self.explainer.encoder = self.encoder
        self.explainer.tokenizer = self.tokenizer
        self.explainer.transformer = self.llm

    def train(self) -> Tuple[R3LExplainer, R3LTrainInfo, Dict[str, Any]]:
        train_loader = DataLoader(self.train_set, batch_size=1, shuffle=True)
        validation_loader = DataLoader(self.validation_set, batch_size=1, shuffle=False)

        encoder_optim = optim.AdamW(
            self.encoder.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(encoder_optim, T_max=self.max_iter)

        loss_history = []
        validation_history = []

        best_validation_loss = None
        early_stopping_counter = 0
        
        if self.logger is not None:
            self.logger.info("Start training")

        n_iter = 0
        for epoch in range(self.max_iter):
            n_iter += 1
            # training
            self.encoder.train()
            history = []
            for episode, explanation in train_loader:
                start = time.time()

                encoder_optim.zero_grad()
                loss = self.explainer.generate_with_loss(
                    prompt=episode,
                    context=None,
                    ground_truth=explanation,
                    think_start=self.llm_think_start,
                    think_end=self.llm_think_end,
                    text_start=self.llm_text_start
                )

                loss.backward()
                total_norm = clip_grad_norm_(self.encoder.parameters(), max_norm=float('inf'), norm_type=2)
                encoder_optim.step()

                if self.logger is not None:
                    self.logger.info(f"TRAIN epoch: {epoch} loss: {loss.item():.4f}, grad_norm: {total_norm.item()}, lr: {encoder_scheduler.get_last_lr()[0]:.6f}, time: {time.time() - start:.2f}s")

                history.append(loss.item())

            if self.logger is not None:
                self.logger.info(f"TRAIN epoch summary: {epoch} loss: {sum(history) / len(history):.4f}")

            encoder_scheduler.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            loss_history.append(history)

            # validation
            self.encoder.eval()
            history = []
            with torch.no_grad():
                for episode, explanation in validation_loader:
                    loss = self.explainer.generate_with_loss(
                        prompt=episode,
                        context=None,
                        ground_truth=explanation,
                        think_start=self.llm_think_start,
                        think_end=self.llm_think_end,
                        text_start=self.llm_text_start
                    )

                    history.append(loss.item())

                if self.logger is not None:
                    self.logger.info(f"VALIDATION epoch: {epoch} loss: {sum(history) / len(history):.4f}")

                validation_history.append(history)

            #early_stopping
            current_loss = sum(history) / len(history)
            if best_validation_loss is None or current_loss < best_validation_loss:
                best_validation_loss = current_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            if early_stopping_counter >= self.early_stopping:
                if self.logger is not None:
                    self.logger.warning("early stopping triggered")
                break

        if self.logger is not None: 
            self.logger.info("training finished")
            
        train_info = R3LTrainInfo(
            pathfile=self.pathfile,
            validation_fraction=self.validation_fraction,
            llm_model_name=self.llm_model_name,
            llm_tokenizer_name=self.llm_tokenizer_name,
            llm_think_start=self.llm_think_start,
            llm_think_end=self.llm_think_end,
            llm_text_start=self.llm_text_start,
            modules=self.modules,
            module_size=self.module_size,
            hidden_size=self.hidden_size,
            projected_size=self.projected_size,
            encoder_device=self.encoder_device,
            llm_device=self.llm_device,
            n_iter=n_iter,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            early_stopping=self.early_stopping,
            seed=self.seed
        )

        additional_info = {
            "loss_history": loss_history,
            "validation_history": validation_history
        }
        
        self.encoder.eval()
        return self.explainer, train_info, additional_info
