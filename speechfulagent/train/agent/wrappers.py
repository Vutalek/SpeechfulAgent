import gymnasium as gym


class RewardWrapper(gym.Wrapper):
    """https://github.com/kondster/DQN-FrozenLake/blob/main/dqn_frozenlake.py"""
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        next_state, reward, is_done, is_trunc, info = self.env.step(action)
        # Modify the reward
        if reward == 0 and not (is_done or is_trunc):
            reward = -0.05  # Increased penalty for each step
        elif reward == 1:
            reward = 5  # Increased reward for reaching the goal
        else:
            reward = -5  # Increased penalty for falling into a holes
        return next_state, reward, is_done, is_trunc, info