import argparse
import logging
logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s]: %(message)s")
logger = logging.getLogger("train")

import gymnasium as gym

from speechfulagent import AgentTrainer

ENVIRONMENT = "FrozenLake-v1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", default=".")

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # env = gym.make(ENVIRONMENT, is_slippery=True)
    env = gym.make("CartPole-v1")
    trainer = AgentTrainer(
        env=env,
        objective=100,
        gamma=0.99,
        gae_lambda=0.95,
        trajectory_size=2049,
        epochs=10,
        eps=0.2,
        batch_size=64,
        learning_rate_actor=0.0003,
        learning_rate_critic=0.0003,
        logger=logger if args.verbose else None
    )
    logger.info("start training")
    agent, env_info, train_info = trainer.train()
    logger.info("saving model")
    agent.save_model(args.model_dir, env=env_info, train=train_info)
