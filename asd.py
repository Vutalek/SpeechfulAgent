from speechfulagent.explainer.model import ExplainerTransformer


inst = ExplainerTransformer(38, 1000, batch_first=True)
print(f"Количество параметров в модели: {sum(p.numel() for p in inst.parameters())}")