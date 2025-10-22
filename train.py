import argparse
import logging
logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s]: %(message)s")
logger = logging.getLogger("train")

import gymnasium as gym

from speechfulagent import Agent, Trainer

ENVIRONMENT = "FrozenLake-v1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", default=".")

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    env = gym.make(ENVIRONMENT, is_slippery=False)
    trainer = Trainer(
        env=env,
        objective=0.9,
        gamma=0.99,
        replay_buffer_size=10000,
        replay_buffer_start_size=10000,
        batch_size=32,
        learning_rate=1e-4,
        sync_target_frames=1000,
        epsilon_decay_last_frame=20000,
        epsilon_decay_start=1.0,
        epsilon_decay_final=0.01,
        logger=logger if args.verbose else None
    )
    logger.info("start training")
    agent, env_info, train_info = trainer.train()
    logger.info("saving model")
    agent.save_model(args.model_dir, env_info, train_info)
