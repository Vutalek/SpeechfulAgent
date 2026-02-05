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

    # env = gym.make(ENVIRONMENT, is_slippery=False)
    env = gym.make("MountainCarContinuous-v0")
    trainer = AgentTrainer(
        env=env,
        objective=80,
        gamma=0.95,
        replay_buffer_size=100000,
        replay_buffer_start_size=1000,
        batch_size=64,
        learning_rate=1e-3,
        sync_target_frames=1000,
        ou_enable=True,
        ou_mu=0.0,
        ou_theta=0.15,
        ou_sigma=0.5,
        ou_epsilon=1.0,
        logger=logger if args.verbose else None
    )
    logger.info("start training")
    agent, env_info, train_info = trainer.train()
    logger.info("saving model")
    agent.save_model(args.model_dir, env=env_info, train=train_info)
