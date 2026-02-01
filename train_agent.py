import argparse
import logging
logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s]: %(message)s")
logger = logging.getLogger("train")

from speechfulagent import AgentTrainer

ENVIRONMENT = "FrozenLake-v1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", default=".")

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    trainer = AgentTrainer(
        env=ENVIRONMENT,
        objective=-4.0,
        gamma=0.95,
        entropy_beta=0.01,
        clip_grad=0.1,
        n_envs=4,
        n_steps=4,
        batch_size=6,
        learning_rate=0.001,
        logger=logger if args.verbose else None
    )
    logger.info("start training")
    agent, env_info, train_info = trainer.train()
    logger.info("saving model")
    agent.save_model(args.model_dir, env=env_info, train=train_info)
