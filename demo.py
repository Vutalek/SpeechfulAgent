import argparse
import logging
logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s]: %(message)s")
logger = logging.getLogger("demo")

import gymnasium as gym

from speechfulagent import SpeechfulAgent


ENVIRONMENT = "FrozenLake-v1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent-dir", default=".")
    parser.add_argument("--agent_version", default="latest")
    parser.add_argument("--explainer-dir", default=".")
    parser.add_argument("--explainer_version", default="latest")

    parser.add_argument("--video", default="videos")

    args = parser.parse_args()

    env = gym.make(ENVIRONMENT, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=args.video, name_prefix=args.version)
    agent = SpeechfulAgent(
        agent_dir=args.agent_dir,
        explainer_dir=args.explainer_dir,
        agent_version=args.agent_version,
        explainer_version=args.explainer_version
    )
    logger.info(f"agent version: {agent.agent.get_version()}")
    logger.info(f"explainer version: {agent.explainer.get_version()}")
    agent.set_environment(env)
    _, _, reward = agent.run()
    env.close()
    logger.info(f"Total reward: {reward}")
