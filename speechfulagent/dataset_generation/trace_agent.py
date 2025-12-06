from typing import Tuple, List, Any

import gymnasium as gym

from speechfulagent.agent import Agent
from speechfulagent.train.replay_buffer import ReplayBuffer


def trace_agent(env: gym.Env, agent: Agent) -> Tuple[ReplayBuffer, List[Any], float]:
    sequence = ReplayBuffer(100)
    states = []
    agent.reset()
    state, _ = env.reset()
    states.append(env.render())
    agent.init_state(state)
    while True:
        exp = agent.step(env)
        sequence.append(exp)
        states.append(env.render())
        if exp.done:
            break
    return sequence, states, agent.total_reward