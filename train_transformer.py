import torch

from speechfulagent.explainer import train

tokenizer, history, model = train(
    "dataset/data.json",
    max_length=32,
    max_iter=1000,
    info_every_epoch=50,
    seed=7070
)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "tokenizer_vocab": tokenizer.vocab,
        "loss_history": history
    },
    "models/explainer.pth"
)