from speechfulagent.explainer.model import ExplainerTransformer


inst = ExplainerTransformer(38, 1000, batch_first=True)
print(f"Количество параметров в модели: {sum(p.numel() for p in inst.parameters())}")

import torch
import torch.nn.functional as F
tail = torch.concat([torch.zeros((1, 5, 38)), torch.ones((1, 3, 38))], dim=1)
sequence = torch.ones((1, 8, 38))
tgt = torch.zeros((1, 1), dtype=torch.int32)
inst.eval()
with torch.no_grad():
    result = inst(tail, sequence, tgt)[0, -1, :]
    next_token_probs, _ = F.softmax(result, dim=0).sort(descending=True)
    print(next_token_probs)
    print(next_token_probs.shape)
    print(next_token_probs.sum())
inst.train()
result = inst(tail, sequence, tgt)
loss = 500 - result.sum()
loss.backward()