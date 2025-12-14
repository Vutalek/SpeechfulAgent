import gymnasium as gym
import torch

from speechfulagent.explainer import Tokenizer, Explainer, ExplainerTransformer, SequenceExplanationsDataset
from speechfulagent.dataset_generation import generate_dataset
from speechfulagent.agent import Agent


ENVIRONMENT = "FrozenLake-v1"

if __name__ == "__main__":
    env = gym.make(ENVIRONMENT, render_mode="rgb_array")
    agent = Agent()
    agent.load_model("models", "latest")

    generate_dataset(env, agent, 10, 1, "testing/1")

    model_info = torch.load("models/explainer.pth")
    tokenizer = Tokenizer()
    tokenizer.set_vocab(model_info["tokenizer_vocab"])

    dataset = SequenceExplanationsDataset("testing/1/data.json", 32, tokenizer, seed=777)

    model = ExplainerTransformer(37, tokenizer.vocab_size(), batch_first=True)
    model.load_state_dict(model_info["model_state_dict"])
    model.eval()

    explainer = Explainer(tokenizer, model)
    seq, tail, expl = dataset[0]

    print(f"Количество объясняемых действий: {dataset.raw_data[0]['tail']+1}")
    for _ in range(10):
        result = explainer.generate(tail.unsqueeze(0), seq.unsqueeze(0), temperature=1.2, top_k=20)
        print(result)