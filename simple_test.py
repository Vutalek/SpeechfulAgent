import torch
import torch.nn.functional as F

from speechfulagent.explainer.transformer.model import ExplainerTransformer


inst = ExplainerTransformer(37, 1000, batch_first=True)
print(f"Количество параметров в модели: {sum(p.numel() for p in inst.parameters())}")

print("Тестирование forward...")
tail = torch.concat([torch.zeros((1, 5, 37)), torch.ones((1, 3, 37))], dim=1)
sequence = torch.ones((1, 8, 37))
tgt = torch.zeros((1, 1), dtype=torch.int32)
inst.eval()
with torch.no_grad():
    result = inst(tail, sequence, tgt)[0, -1, :]
    next_token_probs, _ = F.softmax(result, dim=0).sort(descending=True)
    assert(next_token_probs.shape == (1000,))
    assert(next_token_probs.sum() == 1.0)
print("forward работает!")

print("Тестирование backward...")
inst.train()
result = inst(tail, sequence, tgt)
loss = 500 - result.sum()
loss.backward()
print("backward работает!")