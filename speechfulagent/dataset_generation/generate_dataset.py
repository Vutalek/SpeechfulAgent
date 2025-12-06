import os
import random

import gymnasium as gym

from speechfulagent.agent import Agent
from speechfulagent.dataclasses import Experience
from .trace_agent import trace_agent
from .make_image import make_image
from .make_json import make_json


def generate_dataset(env: gym.Env, agent: Agent, length: int, n: int, save_path: str, logger=None):
    random.seed(7777)
    dataset = []
    for i in range(int(n)):
        seq, frames, total_reward = trace_agent(env, agent)
        if logger:
            logger.info(f"Total reward in {i}: {total_reward}")
        for j, (exp, frame) in enumerate(zip(list(seq.buffer), frames)):
            img = make_image(exp, frame)
            if not os.path.exists(os.path.join(save_path, str(i))):
                os.mkdir(os.path.join(save_path, str(i)))
            img.save(os.path.join(save_path, str(i), f"{str(j)}.png"))
        if len(seq) > length:
            for j in range(n - len(seq)):
                seq.append(Experience(-1, -1, -1, -1, True))
                dataset.append(list(seq.buffer))
        else:
            dataset.append(list(seq.buffer)[:n])
    tails = [random.choice(range(length)) for _ in range(n)]
    json_obj = make_json(dataset, tails)
    with open(os.path.join(save_path, "data.json"), "w") as f:
        f.write(json_obj)
    if logger:
        logger.info(f"Dataset generated.")