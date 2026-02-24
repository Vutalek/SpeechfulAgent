import gymnasium as gym

from speechfulagent.agent import Agent
from speechfulagent.agent import A2C


env = gym.make("FrozenLake-v1")
net = A2C(16, 4)
agent = Agent()
agent.net = net
agent.reset()
state, _ = env.reset()
agent.init_state(state)
print(agent.step(env))