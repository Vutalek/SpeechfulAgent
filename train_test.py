import logging
logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s]: %(message)s")
logger = logging.getLogger("test")

from speechfulagent import AgentTrainer


if __name__ == "__main__":
    trainer = AgentTrainer(
        env="CartPole-v1",
        objective=300.0,
        gamma=0.95,
        entropy_beta=0.01,
        clip_grad=0.1,
        n_envs=4,
        n_steps=4,
        batch_size=2,
        learning_rate=0.001,
        logger=logger
    )
    trainer.train()