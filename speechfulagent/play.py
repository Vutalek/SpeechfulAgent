import gymnasium as gym

from speechfulagent.agent import Agent


def play(env: gym.Env, agent: Agent):
    agent.play_episode(env)