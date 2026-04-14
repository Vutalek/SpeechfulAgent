import os
import json
from typing import List

import gymnasium as gym

from speechfulagent.dataclasses import *
from speechfulagent.agent import BaseAgent, A2CAgent
from speechfulagent.agent.net import DiscreteA2C

SAVE_DIR = "episodes_bad"


def play_episode(agent: BaseAgent) -> List[Experience]:
    episode = []
    agent.reset()
    while True:
        exp = agent.step()
        episode.append(exp)
        if exp.done:
            break
    return episode

def episode_to_json(filename: str, episode: List[Experience]) -> bool:
    dicts = [exp.dict() for exp in episode]
    last_exp = {
        "state": dicts[-1]["next_state"],
        "action": None,
        "reward": None,
        "done": True
    }
    for d in dicts:
        del d["next_state"]
    dicts.append(last_exp)
    try:
        with open(filename, "wt") as f:
            json.dump(dicts, f)
    except Exception as e:
        print(f"Exception occured while saving episode: {e}")
        return False
    return True
    
if __name__ == "__main__":
    os.makedirs(SAVE_DIR + "/data", exist_ok=True)
    episodes_info = {
        "save_dir": SAVE_DIR,
        "episodes": []
    }

    env = gym.make("FrozenLake-v1")
    agent = A2CAgent(env)
    # agent.load_model("agent_models")
    agent.set_model(DiscreteA2C(16, 4))
    agent.eval()

    counter = 0
    name_format = SAVE_DIR + "/data/episode{}.json"
    for i in range(20):
        episode = play_episode(agent)
        filename = name_format.format(i)
        episode_to_json(filename, episode)
        episodes_info["episodes"].append(
            {
                "filename": filename,
                "length": len(episode),
                "total_reward": agent.total_reward
            }
        )
    with open(SAVE_DIR + "/metadata.json", "wt") as f:
        json.dump(episodes_info, f)
