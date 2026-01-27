from typing import Tuple, List
import time

import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from .dataset import SequenceExplanationsDataset
from speechfulagent.explainer.preprocessing import Tokenizer
from speechfulagent.explainer.transformer import ExplainerTransformer
from speechfulagent.dataclasses import *

class ExplainerTrainer:
    def __init__(
        self,
        pathfile: str,
        d_state: int,
        tgt_vocab_size: int,
        d_hidden: int=512,
        nhead: int=8,
        num_decoder_layers: int=6,
        dim_feedforward: int=2048,
        max_len_pe: int=500,
        dropout: float=0.1,
        batch_first: bool=False,
        bias: bool=True,
        learning_rate: float=0.001,
        batch_size: int=4,
        max_length: int=32,
        max_iter: int=10,
        info_every_epoch: int=10,
        seed: int=7070
    ):
        self.pathfile = pathfile
        self.dataset = SequenceExplanationsDataset(pathfile, max_length, seed=seed)
        self.max_length = max_length
        self.tokenizer = self.dataset.tokenizer
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        self.model = ExplainerTransformer(
            d_state,
            tgt_vocab_size,
            d_hidden,
            nhead,
            num_decoder_layers,
            dim_feedforward,
            max_len_pe,
            dropout,
            batch_first,
            bias
        )

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.criterion = CrossEntropyLoss(ignore_index=self.tokenizer.special_tokens["<PAD>"])
        self.train_loss_history = []

        self.max_iter = max_iter
        self.info_every_epoch = info_every_epoch

    def train(self) -> Tuple[Tokenizer, List[float], ExplainerTransformer, ExplainerTrainInfo]:
        times = []
        print("======= START OF TRAINING =======")
        for epoch in range(1, self.max_iter+1):
            self.model.train()
            epoch_loss = 0.0
            t1 = time.time()
            for seq, tail, expl in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(tail, seq, expl)
                loss = self.criterion(output.reshape(-1, output.size(-1)), expl.reshape(-1))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            t2 = time.time()

            times.append(t2-t1)
            if epoch % self.info_every_epoch == 0:
                print(f"======= EPOCH {epoch} ".ljust(33, '='))
                print(f"Loss: {epoch_loss}")
                print(f"Avg. Time: {np.mean(times)}")
                times = []
                
            self.train_loss_history.append(epoch_loss)
        print("======= END OF TRAINING =========")
        train_info = ExplainerTrainInfo(
            self.pathfile,
            self.max_length,
            self.max_iter,
            self.batch_size,
            self.learning_rate
        )
        return self.tokenizer, self.train_loss_history, self.model, train_info