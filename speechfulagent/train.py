import gymnasium as gym
from torch.utils.tensorboard.writer import SummaryWriter

from speechfulagent.agent import Agent


def train(
    agent: Agent,
    test_env: gym.Env,
    test_episodes: int,
    objective: float,
    logger = None
):
    writer = SummaryWriter()
    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        state, action, reward, next_state = agent.sample_env()
        agent.value_update(state, action, reward, next_state)

        test_reward = 0.0
        for _ in range(test_episodes):
            test_reward += agent.play_episode(test_env)
        test_reward /= test_episodes
        writer.add_scalar("reward", test_reward, iter_no)
        if test_reward > best_reward:
            if logger:
                logger.info("%d: Best test reward updated %.3f -> %.3f" % (iter_no, best_reward, test_reward))
            best_reward = test_reward
        if test_reward > objective:
            if logger:
                logger.info("Solved in %d iterations!" % iter_no)
            break
    writer.close()
