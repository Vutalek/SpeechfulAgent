import gymnasium as gym

from speechfulagent.agent import Agent


def play(env: gym.Env, agent: Agent) -> float:
    agent.reset()
    state, _ = env.reset()
    agent.init_state(state)
    while True:
        exp = agent.step(env)
        if exp.done:
            break
    return agent.total_reward