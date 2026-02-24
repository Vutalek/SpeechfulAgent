import gymnasium as gym

from speechfulagent.agent import Actor, Critic, Agent


env = gym.make("FrozenLake-v1")
actor = Actor(env.observation_space.n, env.action_space.n)
critic = Critic(env.observation_space.n, env.action_space.n)
agent = Agent(ou_enable=True)

agent.actor = actor
agent.critic = critic
state, _ = env.reset()
agent.reset()
agent.init_state(state)

print(agent.step(env))
print(agent.actions_state)