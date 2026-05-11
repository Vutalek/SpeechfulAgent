import os
import argparse
import logging
logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s]: %(message)s")
logger = logging.getLogger("demo")

import gymnasium as gym
from dotenv import load_dotenv
load_dotenv()

from speechfulagent import SpeechfulAgent
from speechfulagent.agent import A2CAgent
from speechfulagent.explainer import R3LExplainer, OpenaiExplainer


ENVIRONMENT = "FrozenLake-v1"
# ENVIRONMENT = "MountainCarContinuous-v0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent-dir", default=".")
    parser.add_argument("--agent-version", default="latest")
    parser.add_argument("--explainer-dir", default=".")
    parser.add_argument("--explainer-version", default="latest")

    parser.add_argument("--video", default="videos")

    args = parser.parse_args()

    env = gym.make(ENVIRONMENT, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder=args.video, 
        name_prefix=f"{args.agent_version}-{args.explainer_version}"
    )
    rl_agent = A2CAgent(env)
    rl_agent.load_model(args.agent_dir, args.agent_version)
    logger.info(f"agent version: {rl_agent.get_version()}")

    # explainer = R3LExplainer()
    # explainer.load_model(args.explainer_dir, args.explainer_version)
    # logger.info(f"explainer version: {explainer.get_version()}")
    explainer = OpenaiExplainer(
        "prompt.txt",
        api_key=os.environ.get("YA_API_KEY", ''),
        project=None,
        base_url="https://llm.api.cloud.yandex.net/v1",
        model=f"gpt://{os.environ.get('YA_FOLDER_ID', '')}/yandexgpt-5.1/latest"
    )

    agent = SpeechfulAgent(
        max_tokens=128,
        frequency=10,
        temperature=1
    )
    agent.set_agent(rl_agent)
    agent.set_explainer(explainer)
    
    agent.reset()
    _, _, reward = agent.run(need_print=True)
    env.close()
    logger.info(f"Total reward: {reward}")
