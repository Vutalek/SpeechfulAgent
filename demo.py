import argparse
import logging
logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s]: %(message)s")
logger = logging.getLogger("demo")

import gymnasium as gym

from speechfulagent import Agent, Trainer, play


ENVIRONMENT = "FrozenLake-v1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("script", choices=["train", "play"])

    parser.add_argument("--gamma", default=0.9)

    parser.add_argument("--save-dir", default=".")
    parser.add_argument("--version", default="latest")

    parser.add_argument("--objective", default=0.8)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--video", default="videos")

    args = parser.parse_args()

    if args.script == "train":
        env = gym.make(ENVIRONMENT)
        agent = Agent(env, float(args.gamma), float(args.alpha))
        logger.info("start training")
        if args.verbose:
            train(agent, gym.make(ENVIRONMENT), int(args.test_episodes), float(args.objective), logger)
        else:
            train(agent, gym.make(ENVIRONMENT), int(args.test_episodes), float(args.objective))
        logger.info("saving model")
        agent.save_model(args.save_dir)
    if args.script == "play":
        env = gym.make(ENVIRONMENT, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder=args.video, name_prefix=f"{args.version}")
        agent = Agent(env, float(args.gamma), float(args.alpha))
        agent.load_model(args.save_dir, args.version)
        logger.info(f"model version: {agent.get_version()}")
        play(env, agent)
        env.close()
