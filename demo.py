import argparse
import logging
logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s]: %(message)s")
logger = logging.getLogger("demo")

import gymnasium as gym

from speechfulagent import Agent, play


ENVIRONMENT = "FrozenLake-v1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--version", default="latest")

    parser.add_argument("--video", default="videos")

    args = parser.parse_args()

    env = gym.make(ENVIRONMENT, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=args.video, name_prefix=args.version)
    agent = Agent()
    agent.load_model(args.model_dir, args.version)
    logger.info(f"model version: {agent.get_version()}")
    play(env, agent)
    env.close()
    logger.info(f"Total reward: {agent.total_reward}")
