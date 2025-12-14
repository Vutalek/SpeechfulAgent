from typing import Tuple, List
import time

import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from speechfulagent.explainer.preprocessing import Tokenizer
from speechfulagent.explainer.transformer import ExplainerTransformer
from speechfulagent.explainer.training.dataset import SequenceExplanationsDataset


def train(
        pathfile: str,
        max_length: int = 32,
        max_iter: int = 100,
        info_every_epoch: int = 10,
        seed: int = 7070
) -> Tuple[Tokenizer, List[float], ExplainerTransformer]:
    dataset = SequenceExplanationsDataset(pathfile, max_length, seed=seed)
    tokenizer = dataset.tokenizer
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = ExplainerTransformer(37, tokenizer.vocab_size(), batch_first=True)
    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss(ignore_index=tokenizer.special_tokens["<PAD>"])
    train_loss_history = []

    times = []
    print("======= START OF TRAINING =======")
    for epoch in range(1, max_iter+1):
        model.train()
        epoch_loss = 0.0
        t1 = time.time()
        for seq, tail, expl in dataloader:
            optimizer.zero_grad()
            output = model(tail, seq, expl)
            loss = criterion(output.reshape(-1, output.size(-1)), expl.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        t2 = time.time()

        times.append(t2-t1)
        if epoch % info_every_epoch == 0:
            print(f"======= EPOCH {epoch} ".ljust(33, '='))
            print(f"Loss: {epoch_loss}")
            print(f"Avg. Time: {np.mean(times)}")
            times = []
            
        train_loss_history.append(epoch_loss)
    print("======= END OF TRAINING =========")
    return tokenizer, train_loss_history, model