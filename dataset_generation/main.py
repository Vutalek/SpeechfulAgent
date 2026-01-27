import argparse
import logging
logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s]: %(message)s")
logger = logging.getLogger("demo")

import gymnasium as gym

from speechfulagent import Agent
from .generate_dataset import generate_dataset


ENVIRONMENT = "FrozenLake-v1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--version", default="latest")

    parser.add_argument("--length", default="10")
    parser.add_argument("--save-dir", default=".")
    parser.add_argument("--n", default="50")

    args = parser.parse_args()

    env = gym.make(ENVIRONMENT, render_mode="rgb_array")
    agent = Agent()
    agent.load_model(args.model_dir, args.version)
    logger.info(f"model version: {agent.get_version()}")
    
    generate_dataset(env, agent, int(args.length), int(args.n), args.save_dir, logger)

    env.close()